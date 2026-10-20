//! Live typing: types a streaming transcription into the focused app while
//! the user is still speaking.
//!
//! The stream's `committed` text is append-only, so each new piece of it is
//! typed at the cursor. Commits arrive in bursts of a few words about once a
//! second, so new text is typed at a steady pace spread over the expected gap
//! to the next commit rather than all at once. The same filler removal and
//! voice commands as the final transcript are applied as it goes, holding back
//! a trailing word that may start a command ("new" before "line"). Line breaks
//! are typed right away, so the user sees that "new line" worked.
//!
//! Typing pauses while a modifier key is held (a held Ctrl would turn letters
//! into shortcuts) and while another window is focused. When the user moves
//! to another app and it keeps the focus for [`FOCUS_SETTLE`], typing follows
//! it: text spoken from then on goes there, starting with a capital, and text
//! still untyped at the switch is left out. When the recording stops,
//! [`finish`] reconciles what was typed with the final transcript: missing
//! text is appended, and a rewritten tail of up to [`MAX_FINAL_FIX`]
//! characters is backspaced and retyped.
//!
//! One session runs at a time, on its own worker thread. Dropping the
//! session's sender (a new [`begin`], or [`cancel`]) ends the worker.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use enigo::{Direction, Enigo, Key, Keyboard};
use log::{debug, info, warn};
use tauri::{AppHandle, Manager};
use tauri_plugin_clipboard_manager::ClipboardExt;

use crate::audio_toolkit::{
    apply_voice_commands, normalize_transcription_output, remove_filler_words,
    trailing_voice_command_start, OutputLanguageEvidence,
};
use crate::input::{self, EnigoState};
use crate::settings::{get_settings, AppSettings, ClipboardHandling};

/// How often the worker types the next characters and re-checks modifiers and
/// focus while text is pending.
const POLL_INTERVAL: Duration = Duration::from_millis(15);
/// Assumed gap between commits until one has been measured.
const INITIAL_COMMIT_GAP: Duration = Duration::from_millis(1000);
/// Gaps longer than this are pauses in speech, not the model's cadence.
const MAX_COMMIT_GAP: Duration = Duration::from_secs(3);
/// Share of the expected gap to the next commit spent typing a commit's text,
/// so typing usually finishes just before more text arrives.
const PACE_SHARE: f64 = 0.8;
/// Slowest typing pace, in characters per second.
const MIN_CHARS_PER_SEC: f64 = 20.0;
/// Largest committed-text revision fixed while speaking. Committed text is
/// append-only, so this only guards against an unexpected rewrite.
const MAX_LIVE_FIX: usize = 24;
/// Largest rewritten tail fixed when the recording stops (custom words,
/// filler removal, words the live pass misheard).
const MAX_FINAL_FIX: usize = 400;
/// How long [`finish`] waits for modifiers to be released and the target
/// window to be focused before giving up on the remaining text.
const FINISH_READY_TIMEOUT: Duration = Duration::from_secs(3);
/// Upper bound on [`finish`] blocking its caller.
const FINISH_REPLY_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a newly focused window must keep the focus before typing moves
/// to it, so windows passed through on the way (Alt+Tab) do not count.
const FOCUS_SETTLE: Duration = Duration::from_millis(500);

enum Msg {
    Committed(String),
    Finish {
        final_text: String,
        reply: Sender<bool>,
    },
}

/// Managed state holding the active session's channel, if any.
#[derive(Default)]
pub struct LiveTyper {
    session: Mutex<Option<Sender<Msg>>>,
}

impl LiveTyper {
    pub fn new() -> Self {
        Self::default()
    }

    fn send(&self, msg: Msg) -> bool {
        let session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        session.as_ref().is_some_and(|tx| tx.send(msg).is_ok())
    }

    fn replace(&self, tx: Option<Sender<Msg>>) {
        *self.session.lock().unwrap_or_else(|e| e.into_inner()) = tx;
    }

    fn take(&self) -> Option<Sender<Msg>> {
        self.session.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// Start live typing for a recording, targeting the window focused now.
/// Replaces (and so ends) any previous session.
pub fn begin(app: &AppHandle) {
    let Some(typer) = app.try_state::<LiveTyper>() else {
        return;
    };
    let (tx, rx) = mpsc::channel();
    let target = platform::foreground_window();
    let app = app.clone();
    typer.replace(Some(tx));
    info!("Live typing started");
    thread::spawn(move || Worker::new(app, target).run(rx));
}

/// Feed the stream's latest committed text. No-op without a session.
pub fn on_committed(app: &AppHandle, committed: &str) {
    if let Some(typer) = app.try_state::<LiveTyper>() {
        typer.send(Msg::Committed(committed.to_string()));
    }
}

/// End the session without typing anything more.
pub fn cancel(app: &AppHandle) {
    if let Some(typer) = app.try_state::<LiveTyper>() {
        if typer.take().is_some() {
            debug!("Live typing cancelled");
        }
    }
}

/// Reconcile the typed text with the final transcript and end the session.
///
/// Returns `true` when live typing has handled the output, including the
/// trailing space, auto-submit and clipboard settings. Returns `false` when
/// there was no session; the caller should then paste `final_text` the usual
/// way.
pub fn finish(app: &AppHandle, final_text: &str) -> bool {
    let Some(typer) = app.try_state::<LiveTyper>() else {
        return false;
    };
    let Some(tx) = typer.take() else {
        return false;
    };
    let (reply_tx, reply_rx) = mpsc::channel();
    let msg = Msg::Finish {
        final_text: final_text.to_string(),
        reply: reply_tx,
    };
    if tx.send(msg).is_err() {
        return false;
    }
    reply_rx.recv_timeout(FINISH_REPLY_TIMEOUT).unwrap_or(false)
}

struct Worker {
    app: AppHandle,
    settings: AppSettings,
    language: OutputLanguageEvidence,
    target: Option<isize>,
    /// A newly focused window typing may move to, and since when it has had
    /// the focus.
    candidate: Option<(isize, Instant)>,
    /// The committed text when typing last moved to another window; it
    /// belongs to earlier windows. `None` while in the first window.
    switched_after: Option<String>,
    /// Text typed into the current window so far.
    typed: String,
    /// Latest committed text from the stream, cleaned up like the final
    /// transcript.
    committed: String,
    /// Set when the committed text diverged beyond what is safe to fix live;
    /// the rest is left for [`finish`].
    stalled: bool,
    /// When the session started, for timing logs.
    started: Instant,
    last_commit: Option<Instant>,
    /// Smoothed gap between commits.
    commit_gap: Duration,
    /// Current typing pace, in characters per second.
    pace: f64,
    /// Characters owed by the pace but not typed yet.
    budget: f64,
    last_tick: Instant,
    /// Why typing is currently waiting, logged once per wait.
    waiting: Option<&'static str>,
}

impl Worker {
    fn new(app: AppHandle, target: Option<isize>) -> Self {
        let settings = get_settings(&app);
        let language = match settings.selected_language.as_str() {
            "auto" => OutputLanguageEvidence::Unknown,
            lang => OutputLanguageEvidence::UserSelected(lang.to_string()),
        };
        Self {
            app,
            settings,
            language,
            target,
            candidate: None,
            switched_after: None,
            typed: String::new(),
            committed: String::new(),
            stalled: false,
            started: Instant::now(),
            last_commit: None,
            commit_gap: INITIAL_COMMIT_GAP,
            pace: MIN_CHARS_PER_SEC,
            budget: 0.0,
            last_tick: Instant::now(),
            waiting: None,
        }
    }

    fn run(mut self, rx: Receiver<Msg>) {
        loop {
            match rx.recv_timeout(POLL_INTERVAL) {
                Ok(Msg::Committed(text)) => {
                    self.on_committed(&text);
                    self.type_pending();
                }
                Ok(Msg::Finish { final_text, reply }) => {
                    let _ = reply.send(self.finish(&final_text));
                    return;
                }
                Err(RecvTimeoutError::Timeout) => self.type_pending(),
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }

    /// Move typing to a newly focused window once it has kept the focus for
    /// [`FOCUS_SETTLE`]. Called every poll, whether or not text is pending.
    fn follow_focus(&mut self) {
        let foreground = platform::foreground_window();
        if foreground == self.target {
            self.candidate = None;
            return;
        }
        let Some(window) = foreground.filter(|&w| platform::can_type_into(w)) else {
            self.candidate = None;
            return;
        };
        match self.candidate {
            Some((w, since)) if w == window => {
                if since.elapsed() >= FOCUS_SETTLE {
                    self.retarget(window);
                }
            }
            _ => self.candidate = Some((window, Instant::now())),
        }
    }

    /// Start typing into `window`: from here on it gets the text committed
    /// after this point.
    fn retarget(&mut self, window: isize) {
        let untyped = self.pending().1;
        info!(
            "Live typing: moved to the focused window (class '{}', was '{}'); \
             {} untyped chars from before the switch left out",
            platform::window_class(Some(window)),
            platform::window_class(self.target),
            untyped
        );
        self.target = Some(window);
        self.candidate = None;
        self.switched_after = Some(self.committed.clone());
        self.typed.clear();
        self.stalled = false;
        self.budget = 0.0;
    }

    /// The text that belongs in the current window: all the committed text,
    /// or after a switch, what was committed since.
    fn goal(&self) -> String {
        match &self.switched_after {
            None => self.committed.clone(),
            Some(before) => {
                let rest = self
                    .committed
                    .strip_prefix(before.as_str())
                    .or_else(|| self.committed.get(before.len()..))
                    .unwrap_or("");
                window_start(rest)
            }
        }
    }

    /// The part of the final transcript that belongs in the current window.
    /// The final transcript usually starts with the text committed before the
    /// switch; if the model rewrote that, the text typed here is looked up in
    /// it instead. `None` if neither is found.
    fn final_goal(&self, final_text: &str) -> Option<String> {
        let Some(before) = &self.switched_after else {
            return Some(final_text.to_string());
        };
        if let Some(rest) = final_text.strip_prefix(before.as_str()) {
            return Some(window_start(rest));
        }
        // Skip the first character, which may have been capitalized here.
        let first = self.typed.chars().next()?;
        let typed_rest = &self.typed[first.len_utf8()..];
        // Too short a piece could be found in the wrong place.
        if typed_rest.chars().count() < 8 {
            return None;
        }
        let found = final_text.rfind(typed_rest)?;
        let start = final_text[..found]
            .char_indices()
            .next_back()
            .map_or(found, |(i, _)| i);
        Some(window_start(&final_text[start..]))
    }

    fn ready(&mut self) -> bool {
        let foreground = platform::foreground_window();
        let blocked = if platform::modifiers_held() {
            Some("a modifier key is held")
        } else if foreground != self.target {
            Some("another window is focused")
        } else {
            None
        };
        if blocked != self.waiting {
            match blocked {
                Some(reason) => debug!(
                    "Live typing waiting: {} (focused window class '{}', target '{}')",
                    reason,
                    platform::window_class(foreground),
                    platform::window_class(self.target)
                ),
                None => debug!("Live typing resumed"),
            }
            self.waiting = blocked;
        }
        blocked.is_none()
    }

    /// The committed text as it should appear on screen: cleaned up like the
    /// final transcript, minus a trailing word that may start a voice command.
    fn clean(&self, committed: &str) -> String {
        let text = remove_filler_words(
            committed,
            &self.language,
            &self.settings.custom_filler_words,
            self.settings.filler_word_removal_enabled,
        );
        let text = normalize_transcription_output(&text);
        if !self.settings.voice_commands {
            return text;
        }
        let mut text = apply_voice_commands(&text);
        text.truncate(trailing_voice_command_start(&text));
        text
    }

    /// Take a new committed text and set the pace to type its new part over
    /// most of the expected gap to the next commit.
    fn on_committed(&mut self, raw: &str) {
        let committed = self.clean(raw);
        // Updates that only change the tentative tail say nothing about pace.
        if committed == self.committed {
            return;
        }

        let now = Instant::now();
        let behind = self.pending().1;
        let before = self.committed.chars().count();
        self.committed = committed;
        let added = self.committed.chars().count().saturating_sub(before);
        let gap = self.last_commit.replace(now).map(|last| now - last);
        if let Some(gap) = gap.filter(|gap| *gap < MAX_COMMIT_GAP) {
            self.commit_gap = self.commit_gap.mul_f64(0.7) + gap.mul_f64(0.3);
        }
        let pending = self.pending().1;
        self.pace = (pending as f64 / (self.commit_gap.as_secs_f64() * PACE_SHARE))
            .max(MIN_CHARS_PER_SEC);
        debug!(
            "Live typing: commit at {:.2}s, {} ms after the last, +{} chars, {} still untyped from before, pace {:.0} chars/s",
            (now - self.started).as_secs_f64(),
            gap.map_or(0, |gap| gap.as_millis()),
            added,
            behind,
            self.pace
        );
    }

    /// Byte length of the typed text still matching the committed text, and
    /// how many committed characters come after it.
    fn pending(&self) -> (usize, usize) {
        let goal = self.goal();
        let common = common_prefix_bytes(&self.typed, &goal);
        (common, goal[common..].chars().count())
    }

    /// Type the next few committed characters the pace allows, if it is safe
    /// now. A rewrite of typed text is fixed right away.
    fn type_pending(&mut self) {
        let now = Instant::now();
        let elapsed = now - self.last_tick;
        self.last_tick = now;

        self.follow_focus();
        let goal = self.goal();
        if self.stalled || goal == self.typed {
            self.budget = 0.0;
            return;
        }
        if !self.ready() {
            return;
        }
        let common = common_prefix_bytes(&self.typed, &goal);
        let pending = goal[common..].chars().count();
        let rewrite = common < self.typed.len();
        self.budget = (self.budget + self.pace * elapsed.as_secs_f64()).min(pending as f64);
        let count = self.budget as usize;
        if count == 0 && !rewrite {
            return;
        }

        let end = goal[common..]
            .char_indices()
            .nth(count)
            .map_or(goal.len(), |(i, _)| common + i);
        if self.converge(&goal[..end], MAX_LIVE_FIX) {
            self.budget -= count as f64;
            if self.typed == goal {
                debug!(
                    "Live typing: caught up at {:.2}s",
                    self.started.elapsed().as_secs_f64()
                );
            }
        } else {
            warn!("Live typing: committed text was rewritten; waiting for the final transcript");
            self.stalled = true;
        }
    }

    /// Edit the typed text into `target`: backspace the differing tail (if it
    /// is at most `max_fix` characters) and type the rest. Returns `false`
    /// without touching anything when the tail is longer than that.
    fn converge(&mut self, target: &str, max_fix: usize) -> bool {
        let common = common_prefix_bytes(&self.typed, target);
        let remove = self.typed[common..].chars().count();
        if remove > max_fix {
            return false;
        }
        let insert = &target[common..];
        if remove > 0 {
            debug!(
                "Live typing: correcting {} typed chars '{}' to '{}'",
                remove,
                shape(&self.typed[common..]),
                shape(insert)
            );
        }
        let result = with_enigo(&self.app, |enigo| {
            for _ in 0..remove {
                enigo
                    .key(Key::Backspace, Direction::Click)
                    .map_err(|e| format!("Failed to send Backspace: {}", e))?;
            }
            // Line breaks are typed as Enter; apps differ in what they do
            // with a typed line feed character.
            for (i, line) in insert.split('\n').enumerate() {
                if i > 0 {
                    enigo
                        .key(Key::Return, Direction::Click)
                        .map_err(|e| format!("Failed to send Enter: {}", e))?;
                }
                if !line.is_empty() {
                    input::paste_text_direct(enigo, line)?;
                }
            }
            Ok(())
        });
        match result {
            Ok(()) => {
                self.typed.truncate(common);
                self.typed.push_str(insert);
            }
            Err(e) => {
                warn!("Live typing failed: {}", e);
                self.stalled = true;
            }
        }
        true
    }

    /// Type whatever the final transcript adds, even when nothing was typed
    /// live (a short phrase), so the output never depends on the paste method.
    fn finish(&mut self, final_text: &str) -> bool {
        let settings = self.settings.clone();
        let mut final_text = final_text.trim_start().to_string();
        if settings.append_trailing_space {
            final_text.push(' ');
        }

        let deadline = Instant::now() + FINISH_READY_TIMEOUT;
        self.follow_focus();
        while !self.ready() && Instant::now() < deadline {
            thread::sleep(POLL_INTERVAL);
            self.follow_focus();
        }
        let goal = self.final_goal(&final_text).unwrap_or_else(|| {
            warn!("Live typing: final transcript not matched after a window switch; keeping the typed text");
            self.typed.clone()
        });
        if !self.ready() {
            if goal == self.typed {
                debug!("Live typing finished; nothing left to type");
            } else {
                warn!(
                    "Live typing: {}; leaving the rest on the clipboard",
                    self.waiting.unwrap_or("not ready")
                );
                let _ = self.app.clipboard().write_text(&final_text);
            }
            return true;
        }

        if !self.converge(&goal, MAX_FINAL_FIX) {
            info!(
                "Live typing: final transcript differs by more than {} characters; \
                 keeping the typed text",
                MAX_FINAL_FIX
            );
        }

        if settings.auto_submit {
            thread::sleep(Duration::from_millis(50));
            if let Err(e) = with_enigo(&self.app, |enigo| {
                crate::clipboard::send_return_key(enigo, settings.auto_submit_key)
            }) {
                warn!("Live typing: auto-submit failed: {}", e);
            }
        }
        if settings.clipboard_handling == ClipboardHandling::CopyToClipboard {
            let _ = self.app.clipboard().write_text(&final_text);
        }
        info!("Live typing finished");
        true
    }
}

fn with_enigo(
    app: &AppHandle,
    f: impl FnOnce(&mut Enigo) -> Result<(), String>,
) -> Result<(), String> {
    let state = app
        .try_state::<EnigoState>()
        .ok_or("Enigo state not initialized")?;
    let mut enigo = state
        .0
        .lock()
        .map_err(|e| format!("Failed to lock Enigo: {}", e))?;
    f(&mut enigo)
}

/// Text that opens the typing in a newly focused window: leading spaces
/// dropped and the first letter capitalized.
fn window_start(text: &str) -> String {
    let text = text.trim_start_matches([' ', '\t']);
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Length in bytes of the longest common prefix of `a` and `b`, on a char
/// boundary of both.
fn common_prefix_bytes(a: &str, b: &str) -> usize {
    a.char_indices()
        .zip(b.chars())
        .find(|((_, ca), cb)| ca != cb)
        .map(|((i, _), _)| i)
        .unwrap_or_else(|| a.len().min(b.len()))
}

/// The layout of `text` without its words, for logs: letters become `a`/`A`,
/// digits `0`, line breaks `⏎`; spaces and punctuation are kept.
fn shape(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' => '⏎',
            c if c.is_uppercase() => 'A',
            c if c.is_alphabetic() => 'a',
            c if c.is_numeric() => '0',
            c => c,
        })
        .collect()
}

#[cfg(windows)]
mod platform {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GetClassNameW, GetForegroundWindow, GetWindowThreadProcessId,
    };

    /// Window classes of the Windows shell (taskbar, desktop, Alt+Tab, Task
    /// View, Start and search), which take the focus but are not for typing.
    const SHELL_CLASSES: &[&str] = &[
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
        "Progman",
        "WorkerW",
        "MultitaskingViewFrame",
        "TaskSwitcherWnd",
        "XamlExplorerHostIslandWindow",
        "ForegroundStaging",
        "Windows.UI.Core.CoreWindow",
        "#32768",
    ];

    /// The foreground window as a plain integer (HWND is not `Send`).
    pub fn foreground_window() -> Option<isize> {
        let hwnd = unsafe { GetForegroundWindow() };
        (!hwnd.0.is_null()).then_some(hwnd.0 as isize)
    }

    /// The window's class name ("Notepad", "Chrome_WidgetWin_1"), for logs.
    pub fn window_class(window: Option<isize>) -> String {
        let Some(window) = window else {
            return "none".to_string();
        };
        let mut name = [0u16; 128];
        let len = unsafe { GetClassNameW(HWND(window as *mut _), &mut name) };
        String::from_utf16_lossy(&name[..len.max(0) as usize])
    }

    /// Whether typing may move to `window`: not one of this app's own
    /// windows (settings, overlay) and not part of the Windows shell.
    pub fn can_type_into(window: isize) -> bool {
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(HWND(window as *mut _), Some(&mut pid)) };
        pid != std::process::id() && !SHELL_CLASSES.contains(&window_class(Some(window)).as_str())
    }

    pub fn modifiers_held() -> bool {
        [VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN]
            .iter()
            .any(|vk| {
                // The high bit is set while the key is down.
                let state = unsafe { GetAsyncKeyState(i32::from(vk.0)) };
                state < 0
            })
    }
}

#[cfg(not(windows))]
mod platform {
    // No focus or modifier tracking off Windows yet: always type.
    pub fn foreground_window() -> Option<isize> {
        None
    }

    pub fn window_class(_window: Option<isize>) -> String {
        "none".to_string()
    }

    pub fn can_type_into(_window: isize) -> bool {
        false
    }

    pub fn modifiers_held() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{common_prefix_bytes, shape, window_start};

    #[test]
    fn window_start_drops_spaces_and_capitalizes() {
        assert_eq!(window_start(" now the code"), "Now the code");
        assert_eq!(window_start("\nnext"), "\nnext");
        assert_eq!(window_start(""), "");
    }

    #[test]
    fn shape_hides_words_but_keeps_layout() {
        assert_eq!(shape("Hi, Tom!\n\nPage 2."), "Aa, Aaa!⏎⏎Aaaa 0.");
    }

    #[test]
    fn common_prefix_of_extension_is_the_shorter_string() {
        assert_eq!(common_prefix_bytes("hello", "hello world"), 5);
        assert_eq!(common_prefix_bytes("hello world", "hello"), 5);
        assert_eq!(common_prefix_bytes("", "abc"), 0);
    }

    #[test]
    fn common_prefix_stops_at_first_difference() {
        assert_eq!(common_prefix_bytes("hello there", "hello their"), 9);
        assert_eq!(common_prefix_bytes("abc", "xbc"), 0);
    }

    #[test]
    fn common_prefix_respects_multibyte_chars() {
        // "é" is two bytes; the prefix must end on a char boundary.
        assert_eq!(common_prefix_bytes("café au", "café ou"), 6);
        assert_eq!(common_prefix_bytes("naïve", "naive"), 2);
    }
}
