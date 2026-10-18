//! Live typing: types a streaming transcription into the focused app while
//! the user is still speaking.
//!
//! The stream's `committed` text is append-only, so each new piece of it is
//! typed at the cursor as it arrives. Typing pauses while a modifier key is
//! held (a held Ctrl would turn letters into shortcuts) and while another
//! window is focused, and resumes once both clear. When the recording stops,
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

use crate::input::{self, EnigoState};
use crate::settings::{get_settings, ClipboardHandling};

/// How often the worker re-checks modifiers and focus while text is pending.
const POLL_INTERVAL: Duration = Duration::from_millis(40);
/// Largest committed-text revision fixed while speaking. Committed text is
/// append-only, so this only guards against an unexpected rewrite.
const MAX_LIVE_FIX: usize = 24;
/// Largest rewritten tail fixed when the recording stops (custom words,
/// filler removal near the end, punctuation).
const MAX_FINAL_FIX: usize = 64;
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
/// there was no session or nothing was typed yet; the caller should then
/// paste `final_text` the usual way.
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
    target: Option<isize>,
    /// Text typed so far this session.
    typed: String,
    /// Latest committed text from the stream, leading whitespace trimmed.
    committed: String,
    /// Set when the committed text diverged beyond what is safe to fix live;
    /// the rest is left for [`finish`].
    stalled: bool,
}

impl Worker {
    fn new(app: AppHandle, target: Option<isize>) -> Self {
        Self {
            app,
            target,
            typed: String::new(),
            committed: String::new(),
            stalled: false,
        }
    }

    fn run(mut self, rx: Receiver<Msg>) {
        loop {
            match rx.recv_timeout(POLL_INTERVAL) {
                Ok(Msg::Committed(text)) => {
                    self.committed = text.trim_start().to_string();
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

    fn ready(&self) -> bool {
        !platform::modifiers_held() && platform::foreground_window() == self.target
    }

    /// Type whatever committed text is not on screen yet, if it is safe now.
    fn type_pending(&mut self) {
        if self.stalled || self.committed == self.typed || !self.ready() {
            return;
        }
        let committed = self.committed.clone();
        if !self.converge(&committed, MAX_LIVE_FIX) {
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
            if !insert.is_empty() {
                input::paste_text_direct(enigo, insert)?;
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

    fn finish(&mut self, final_text: &str) -> bool {
        // Nothing reached the screen: let the normal paste path handle it.
        if self.typed.is_empty() {
            return false;
        }

        let settings = get_settings(&self.app);
        let mut final_text = final_text.trim_start().to_string();
        if settings.append_trailing_space {
            final_text.push(' ');
        }

        let deadline = Instant::now() + FINISH_READY_TIMEOUT;
        while !self.ready() && Instant::now() < deadline {
            thread::sleep(POLL_INTERVAL);
        }
        if !self.ready() {
            warn!("Live typing: target window lost focus; leaving the rest on the clipboard");
            let _ = self.app.clipboard().write_text(&final_text);
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
