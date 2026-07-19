---
log:
2026-07-19: Addressed Windows console review findings. The resize poller now starts from the WebSocket open callback, after `_is_running` becomes true, and forwards the size already read from `CONOUT$` instead of querying a possibly redirected stdout. `raw_mode()` restores every console mode that was successfully changed even when later VT setup fails, and Win32 calls preserve the real last-error value for diagnostics. Console handles are closed for partial initialization as well as normal exit. The live Windows smoke uses an attempt-unique session name, always tries to stop that owned session including when `colab new` fails after persisting it, and returns nonzero if `stop` or the final session audit fails.
2026-07-19: Hardened reconnects to an existing kernel. `jupyter-kernel-client` can return from `start_channels()` after its readiness wait expires even though the WebSocket never opened; `ColabRuntime` now rejects that half-connected client, preserves the already-created kernel id, closes only the failed local connection, and performs the existing bounded startup retry. The live Windows CPU smoke covers two consecutive `exec` invocations against one session.
2026-07-19: Added automation-safe notebook execution controls. `colab exec --cell-title <title>` may be repeated to select uniquely named `# @title` code cells while preserving notebook order; missing or ambiguous titles fail before a kernel connection. `--fail-on-error` stops after the first Jupyter `error` output and returns exit code 1. Both controls are opt-in so existing whole-notebook and fail-open behavior remains compatible.
2026-07-05: Added Windows support to `colab console`. The POSIX `termios`/`tty` imports in `console.py` were unguarded, which crashed the entire CLI on Windows (`import colab_cli.cli` raised `ModuleNotFoundError: No module named 'termios'`). The imports are now platform-guarded (`termios = tty = None` on win32), and a new `colab_cli/_winconsole.py` provides a ctypes-based raw-console path: it saves/restores CONIN$/CONOUT$ modes, enables `ENABLE_VIRTUAL_TERMINAL_INPUT` (disabling echo/line/processed input) and `ENABLE_VIRTUAL_TERMINAL_PROCESSING` on output, and a daemon polling thread replaces `SIGWINCH` (which does not exist on Windows) by sampling `GetConsoleScreenBufferInfo` every 0.25s. The piped-stdin path is unchanged and cross-platform. `colab repl`'s `PromptSession` construction was also deferred from `ColabREPL.__init__` to `run()` so it no longer requires a real console at construction time (prompt_toolkit's `Win32Output` calls `GetConsoleScreenBufferInfo` at init and raised under captured stdout on Windows).
2026-05-07: Fixed `colab console` piped-stdin handling. Previously a piped invocation (e.g. `echo 'cmd' | colab console -s s`) sent the command and then hung indefinitely because the previous EOF handler emitted a bare `\x04` (Ctrl-D), which the remote `tmux`-wrapped bash treats as a literal character rather than a session terminator. The new handler sends `exit\n` (which bash actually exits on) and then closes the websocket from the client side after a short grace period (`PIPED_EOF_GRACE_SECONDS = 0.5s`) so any tail output (bash `logout`, tmux `[exited]`) makes it back to the user. TTY mode is unchanged: real-terminal EOF is left to the remote shell. Verified live: `echo 'echo HELLO' | colab console -s s` now exits in ~1.2s instead of hanging.

2026-05-07: Fixed `print_kitty` (used by `colab exec --output-image` and any image-producing exec) to no-op when `sys.stdout.isatty()` is false. The Kitty Graphics Protocol escape sequence is meaningless when stdout is a file or pipe and was visually corrupting captured output (a multi-KB base64 PNG blob would land in log files, grep targets, or showboat captures). Image bytes are still saved to disk via `handle_image`'s file-write path; only the inline-render attempt is suppressed.

2026-06-04: Bumped the default `--timeout` for `colab exec` from 10s to 30s (and the matching `colab run` default) so brief silent tasks are less likely to hit a premature `TimeoutError`. Explicit `--timeout` overrides are unaffected.
---

# Design: Execution and Interactive Interaction (`repl`, `exec`, `console`)

## Overview
Execution involves sending Python code (or shell commands) to the Jupyter kernel running on the Colab VM and processing the stream of output messages.

## Approach

### 1. REPL (`colab repl`)
- **Transport**: WebSockets (using `websockets` library if allowed, or a custom `http.client` based long-polling implementation if we're strictly stdlib).
- **Communication**: Jupyter Kernel Messaging Protocol.
    - `execute_request`: Send code string.
    - `execute_reply`: Get status.
    - `iopub.stream`: Capture `stdout` and `stderr`.
- **Interactive Mode**: Standard Python `cmd.Cmd` or `code.InteractiveConsole` for local input/output.
- **Piping Support**: Detect `sys.stdin.isatty()`. If not a TTY, read all input and send as a single execution request.

### 2. Execution (`colab exec`)
- **File Handling**:
    - If file path is local: Read content, send as code.
    - If file path is remote: Execute `!python <path>`.
- **Multi-Modal Output**: Handle `display_data` messages (e.g., `image/png`, `text/html`). For the CLI, we'll save images to temporary files and print their paths, or if the terminal supports it (e.g., iTerm2), inline them.
- **Timeout Configuration**: Exposes a `--timeout` flag (default 30s) to allow long-running silent tasks (like model compilation or data downloading) to execute without being prematurely killed.
- **Notebook cell selection**: Repeating `--cell-title` selects code cells by their exact, unique `# @title` value. Selection preserves notebook order rather than command-line order. A missing title, duplicate request, or duplicate title in the notebook is a usage error detected before connecting to the kernel. The option is only valid with `.ipynb` files.
- **Automation failure policy**: Jupyter reports user-code exceptions as output messages, so compatibility mode continues to return success after displaying them. Automated workflows should pass `--fail-on-error`; the command then saves notebook output, stops the runtime connection, and returns exit code 1 after the first error output.

Examples:

```shell
colab exec -s smoke -f workflow.ipynb \
  --cell-title "Environment preflight" \
  --cell-title "Project smoke" \
  --fail-on-error
```

### 3. Console (`colab console`)
- **Implementation**: Connects directly to the backend terminal endpoint (`/colab/tty`) via WebSockets using `websocket-client`.
- **Interactive**: Bypasses the Jupyter kernel entirely to provide a raw, PTY-backed bash session on the Colab VM.
- **Terminal Management**: Configures `sys.stdin` to raw mode using `termios` and `tty`, passing single characters to the socket and writing raw ANSI escape sequences directly to `sys.stdout.buffer`. Hooks into `SIGWINCH` to communicate local terminal dimensions (`cols`/`rows`) to the remote bash environment so output rendering works perfectly during resizing. On Windows (where `termios`/`tty`/`SIGWINCH` do not exist), `colab_cli/_winconsole.py` provides an equivalent ctypes path: it toggles `ENABLE_VIRTUAL_TERMINAL_INPUT` (raw VT input) on `CONIN$` and `ENABLE_VIRTUAL_TERMINAL_PROCESSING` on `CONOUT$`, and a daemon thread polls `GetConsoleScreenBufferInfo` for resize events.
- **Piped stdin**: Detected via `sys.stdin.isatty()`. When piped, the input characters are forwarded one at a time to the remote pty, and on EOF the client sends `exit\n` and then closes the websocket itself after `PIPED_EOF_GRACE_SECONDS` (0.5s) so the user's shell goodbye text drains back. The remote `/colab/tty` endpoint wraps bash in tmux, which intercepts a bare `\x04` as a literal character — that is why we send `exit\n` rather than Ctrl-D.

## Implementation Details
- **Kernel Management**: `ColabRuntime` (from `colab-agent`) already handles message signing and message types.
- **Output Streaming**: Continuous polling or asynchronous message handling to provide real-time output.
- **Piping Example**: `cat script.py | colab exec -s my-session`.

## Testing Strategy
TDD is mandatory for all execution features.

### 1. Mock Kernel Client
- **Test Case**: Verify `ColabRuntime` correctly sends an `execute_request` message over the websocket.
- **Test Case**: Verify `iopub.stream` messages are correctly handled and printed to `stdout` in real-time.
- **Test Case**: Verify `display_data` (specifically `image/png`) triggers the correct local handling (saving or display).
- **Test Case**: Verify `--fail-on-error` returns nonzero, stops the runtime, and does not execute later cells after a Jupyter error output.
- **Test Case**: Verify repeated `--cell-title` executes only uniquely selected cells in notebook order and rejects missing or ambiguous titles before creating a runtime.
- **Test Case**: If the WebSocket readiness wait expires without opening channels, preserve the remote kernel id, discard only the failed local client, and reconnect with a bounded retry.

### 2. TTY and Piping
- **Test Case**: Mock `sys.stdin.isatty()` to verify `colab repl` correctly switches between interactive mode and one-shot piped execution.
- **Test Case**: Verify large piped inputs are handled without buffer overflow or truncation.
- **Test Case**: `colab console` with piped stdin sends `exit\n` and calls `ws.close()` on EOF (regression: previously sent `\x04` only and hung).
- **Test Case**: `colab console` in TTY mode does not synthesize an exit on EOF (the user owns the session lifecycle).
- **Test Case**: `print_kitty` is a no-op when `sys.stdout.isatty()` is false (regression: previously emitted ANSI/base64 into pipes and files).
- **Test Case**: On Windows, the resize poller starts only after the WebSocket open callback and observes `_is_running=True`.
- **Test Case**: On Windows with redirected stdout, the resize poller forwards the dimensions read directly from `CONOUT$`.
- **Test Case**: On Windows, output VT setup failure restores an input mode that was already changed before propagating the error.
- **Test Case**: A failed Win32 console API reports its real nonzero `winerror`.
- **Test Case**: If a kernel startup callback fails after the client was cached, the stopped client is removed and the next access reconnects.
- **Test Case**: If the live Windows smoke's `colab new` returns nonzero, its `finally` block still stops the attempt-unique session without touching a pre-existing fixed-name session.
- **Test Case**: If the smoke succeeds but `colab stop` returns nonzero, the script completes its final session audit and then exits nonzero.

### 3. Live Windows CPU smoke

After local unit tests pass and CLI OAuth is available, run:

```powershell
pwsh -NoProfile -File integration/repro_windows_exec_control/test.ps1
```

The script allocates a CPU runtime only, verifies title selection and fail-on-error through two consecutive CLI connections to the same live Colab kernel, stops the session in `finally`, removes the generated output notebook, and finishes by listing server-side sessions for orphan detection. It is intentionally not part of CI because it consumes an external runtime and requires user OAuth.
