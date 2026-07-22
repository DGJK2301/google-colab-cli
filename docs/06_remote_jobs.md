---
log:
2026-07-21: Raised the bounded default job-control RPC budget to 120 seconds for cold kernel/helper bootstrap. A finite `wait --timeout` is propagated as the shrinking budget for every status and log-tail call, so the internal control timeout cannot be shorter than or outlive the user's wait deadline.
2026-07-20: Added reconnectable remote jobs with persisted argv/spec, atomic status, separate stdout/stderr logs, byte-offset tailing, bounded wait/cancel control calls, runtime boot identity, and process start-token validation.
---

# Design: Reconnectable Remote Jobs

## Motivation

`colab exec` intentionally keeps one Jupyter execution request open and streams
IOPub output. It is appropriate for interactive work, but a local terminal or
network disconnect cannot later reattach to that request. Long training needs a
different lifecycle.

The job commands separate control from work:

```text
short kernel request
  -> detached runner in the Colab VM
  -> argv-based child process
  -> atomic status.json + stdout.log + stderr.log
later CLI process
  -> reconnect to the session kernel
  -> status / byte-offset tail / wait / cancel
```

## Commands

```bash
colab submit -s work --name train --cwd /content/project -- \
  python -u train.py --config configs/run.yaml

colab jobs -s work
colab tail train -s work --stream stdout --offset 0
colab wait train -s work --timeout 21600
# After a previous bounded read, resume each stream without replaying old logs.
colab wait train -s work --stdout-offset 1048576 --stderr-offset 2048
colab cancel train -s work --grace-seconds 10
```

Arguments after `--` remain an argv list. There is no implicit shell
interpolation; pipelines or redirects require an explicit `bash -lc ...`.
Repeatable `--env KEY=VALUE` entries are persisted in `spec.json`, so secrets
should instead use the runtime's normal credential facilities.

## Remote State

The default root is `/content/.colab-cli/jobs/<job-id>/`:

```text
spec.json
runner.py
launcher.json
status.json
stdout.log
stderr.log
```

Status transitions are `queued -> running -> succeeded|failed|cancelled`.
`lost` is used when the runner exits without committing a terminal status, the
runtime boot identity changes, or a persisted PID no longer identifies the
same process start time.

JSON writes use temporary files, `fsync`, and atomic replacement. Windows local
tests add a short bounded retry for destination-sharing races; Colab/Linux uses
the same transaction without that retry path.

## Reconnection Semantics

- `submit` returns after the detached runner is created.
- Closing the original CLI only closes local Jupyter channels; it does not shut
  down the kernel or job.
- A separate CLI invocation can call `jobs`, `tail`, or `wait` using the same
  saved Colab session.
- `tail` reports `next_offset`; `wait` maintains offsets for both streams.
- A local `wait --timeout` exits 124 and leaves the remote job running.
- Cold kernel connection and helper bootstrap use a bounded 120-second control
  budget. For finite waits, each status/tail RPC receives only the remaining
  user deadline; internal defaults cannot silently terminate a longer wait at
  30 seconds or extend a shorter one.
- `cancel` signals the runner process group, waits for a bounded grace period,
  then force-stops it if necessary.

The VM remains the failure boundary. Stopping/reclaiming the Colab runtime ends
the process. A job root on Drive may preserve logs, but runtime identity makes
the stale job `lost`; it cannot resume computation in a new VM.

## Relationship to Other Colab Tools

The official [Colab MCP server](https://github.com/googlecolab/colab-mcp) is a
browser-session proxy for agent-driven notebook interaction. Community tools
such as [better_colab_MCP](https://github.com/404F0X/better_colab_MCP) expose
background commands and chunked file operations through browser/terminal
automation. This CLI reuses the useful lifecycle ideas but keeps its existing
OAuth, runtime-proxy, Jupyter-kernel, and Contents API architecture; it does not
add a browser/CDP dependency or copy the community implementation.

## Testing

- Runtime tests launch a real detached local runner and verify status, logs,
  exit code, byte offsets, path validation, and non-destructive PID probing.
- Client tests verify argv serialization, helper bootstrap, base64 decoding,
  and bounded cancel calls.
- CLI tests verify resource cleanup, wait exit codes, timeout-without-cancel,
  and environment validation.
- The live integration test uses a free CPU runtime, invokes each lifecycle
  command from a separate local process, and releases the runtime in `finally`.
