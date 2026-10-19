//! Live typing: types a streaming transcription into the focused app while
//! the user is still speaking.
//!
//! The stream's `committed` text is append-only, so each new piece of it is
//! typed at the cursor. Commits arrive in bursts of a few words about once a
//! second, so new text is typed at a steady pace spread over the expected gap
//! to the next commit rather than all at once. The same filler removal and
//! voice commands as the final transcript are applied as it goes, holding back
//! a trailing word that may start a command ("new" before "line").
//!
//! Typing pauses while a modifier key is held (a held Ctrl would turn letters
//! into shortcuts) and while another window is focused, and resumes once both
//! clear. When the recording stops,
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
    /// Text typed so far this session.
    typed: String,
    /// Latest committed text from the stream, cleaned up like the final
    /// transcript.
    committed: String,
    /// Set when the committed text diverged beyond what is safe to fix live;
    /// the rest is left for [`finish`].
    stalled: bool,
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
            typed: String::new(),
            committed: String::new(),
            stalled: false,
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

    fn ready(&mut self) -> bool {
        let blocked = if platform::modifiers_held() {
            Some("a modifier key is held")
        } else if platform::foreground_window() != self.target {
            Some("another window is focused")
        } else {
            None
        };
        if blocked != self.waiting {
            match blocked {
                Some(reason) => debug!("Live typing waiting: {}", reason),
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
        self.committed = committed;

        let now = Instant::now();
        if let Some(last) = self.last_commit.replace(now) {
            let gap = now - last;
            if gap < MAX_COMMIT_GAP {
                self.commit_gap = self.commit_gap.mul_f64(0.7) + gap.mul_f64(0.3);
            }
        }
        let pending = self.pending().1;
        self.pace = (pending as f64 / (self.commit_gap.as_secs_f64() * PACE_SHARE))
            .max(MIN_CHARS_PER_SEC);
    }

    /// Byte length of the typed text still matching the committed text, and
    /// how many committed characters come after it.
    fn pending(&self) -> (usize, usize) {
        let common = common_prefix_bytes(&self.typed, &self.committed);
        (common, self.committed[common..].chars().count())
    }

    /// Type the next few committed characters the pace allows, if it is safe
    /// now. A rewrite of typed text is fixed right away.
    fn type_pending(&mut self) {
        let now = Instant::now();
        let elapsed = now - self.last_tick;
        self.last_tick = now;

        if self.stalled || self.committed == self.typed {
            self.budget = 0.0;
            return;
        }
        if !self.ready() {
            return;
        }
        let (common, pending) = self.pending();
        let rewrite = common < self.typed.len();
        self.budget = (self.budget + self.pace * elapsed.as_secs_f64()).min(pending as f64);
        let count = self.budget as usize;
        if count == 0 && !rewrite {
            return;
        }

        let end = self.committed[common..]
            .char_indices()
            .nth(count)
            .map_or(self.committed.len(), |(i, _)| common + i);
        let goal = self.committed[..end].to_string();
        if self.converge(&goal, MAX_LIVE_FIX) {
            self.budget -= count as f64;
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
        while !self.ready() && Instant::now() < deadline {
            thread::sleep(POLL_INTERVAL);
        }
        if !self.ready() {
            if final_text == self.typed {
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

        if !self.converge(&final_text, MAX_FINAL_FIX) {
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

/// Length in bytes of the longest common prefix of `a` and `b`, on a char
/// boundary of both.
fn common_prefix_bytes(a: &str, b: &str) -> usize {
    a.char_indices()
        .zip(b.chars())
        .find(|((_, ca), cb)| ca != cb)
        .map(|((i, _), _)| i)
        .unwrap_or_else(|| a.len().min(b.len()))
}

#[cfg(windows)]
mod platform {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };
    use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

    /// The foreground window as a plain integer (HWND is not `Send`).
    pub fn foreground_window() -> Option<isize> {
        let hwnd = unsafe { GetForegroundWindow() };
        (!hwnd.0.is_null()).then_some(hwnd.0 as isize)
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

    pub fn modifiers_held() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::common_prefix_bytes;

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
