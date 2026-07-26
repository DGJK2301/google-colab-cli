# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The package version is derived from the git tag via `hatch-vcs`; each release
below corresponds to a tag of the same name.

## [0.7.0rc3] - 2026-07-26

### Added

- Add `colab doctor` and the stable `colab.doctor.v1` contract. The default
  path is local-only and audits installation identity, dependency versions,
  OAuth cache metadata/ACL, strict session/settings parsing, keep-alive PIDs,
  and transfer leases. `--network` performs one bounded assignment query and
  never allocates a runtime.
- Add foreground `colab monitor JOB` with resumable local evidence:
  `stdout.log`, `stderr.log`, `job.jsonl`, `resources.jsonl`, `events.jsonl`,
  `monitor_state.json`, and `summary.json`.
- Add `colab attach --endpoint ENDPOINT -s NAME` and `colab.attach.v1` for
  transactionally adopting an assignment that exists on the current account
  but has no usable local state.

### Changed

- Monitor state binds the session endpoint, persisted remote runtime identity,
  and probed boot ID. Re-running the command resumes byte offsets without
  duplicating logs; terminal jobs do not return until their remaining logs have
  been copied locally.
- Resource probe failure records an unavailable sample without terminating the
  remote training process. Local Ctrl+C stops only the monitor and exits 130.
- Attach validates the exact live endpoint, rejects local conflicts, establishes
  a bounded control kernel by default, starts keep-alive, and rolls back local
  state on failure without releasing the remote assignment.
- Add strict session-store reads/writes for recovery operations so corrupted
  local state is surfaced instead of being silently replaced with an empty
  store.

## [0.7.0rc2] - 2026-07-25

### Added

- Add a cross-process `TransferLease` around the entire verified
  inspect/resume/chunk/hash/atomic-finalize lifecycle. Upload identity binds the
  runtime endpoint and normalized remote target, independent of the
  control-plane auth used to reach that runtime. Download identity binds the
  platform-normalized local destination without rewriting resume paths.
- Add process-identity-backed lease metadata with PID, process start token,
  heartbeat, source size/SHA-256, progress, resume offset, retry count, target,
  and partial path. Active metadata is recycled only after the original process
  is proven gone or its PID has been reused.
- Add stable `colab.transfer.v1` JSON for upload and download, including byte
  counts, resume offset, elapsed time, throughput, ETA, retry count, final
  SHA-256, lease evidence, warnings, errors, and exact resume argv.
- Add deterministic Ctrl+C behavior: preserve verified partial state, record an
  `interrupted` history event, print/return a resume command, and exit 130.

### Changed

- Reuse one `requests.Session` per `ContentsClient` while preserving explicit
  per-request connect/read timeout tuples and deterministic close semantics.
- Send transfer progress and diagnostics to stderr while reserving JSON stdout
  for one final document.
- Close the Contents HTTP pool before the remote executor; cleanup failures are
  warnings and never replace the primary transfer result or exception.

## [0.7.0rc1] - 2026-07-22

### Added

- Add stable `colab.sessions.v1`, `colab.status.v1`, and `colab.jobs.v1`
  machine-readable contracts through `sessions --json`, `status --json`, and
  `jobs --json`.
- Add bounded `status -s NAME --probe --json --timeout SEC` observation for one
  explicitly selected existing runtime. The probe reports requested and
  assigned accelerators separately, then gathers GPU, VRAM, utilization,
  temperature, RAM, `/content` disk, runtime boot identity, Python version, and
  elapsed probe time when those sources are available.
- Read Colab's runtime resource endpoint before using one aggregate supplemental
  request on an already-recorded kernel. The observation path never allocates,
  restarts, releases, mounts, installs, or creates a kernel.
- Include persisted job heartbeat, return code, error, stdout size, and stderr
  size in structured job listings.

### Changed

- JSON observation commands use a read-only state join instead of
  `sync_sessions()`, so status collection cannot prune local state, kill a
  keep-alive process, or write history.
- Redirect dependency/auth diagnostics to stderr while keeping JSON stdout to
  exactly one document. Compute-unit fields are explicitly `null` with an
  unavailable reason; this release does not infer account balance or rates.
- Extend the existing Jupyter timeout guard to direct connections to an exact
  recorded kernel ID used by read-only observation.
- Enable Google Auth's Cloud SDK reauthentication flow for the bundled OAuth
  client, request `accounts.reauth`, cache the resulting RAPT token, and keep
  custom OAuth clients and ADC on their existing scope behavior.

## [0.6.0.post1] - 2026-07-21

### Fixed

- Pin the published Colab Jupyter transport to commit
  `f18e982c3265df5e923aa9def101ab3fd737e139`; add compatible dependency floors
  and a lazy dependency-diagnostic entry point.
- Prevent finite quiet execution timeouts from entering a local CPU busy loop;
  allow one queued/event-boundary message without permitting continuous output
  to extend the wall-clock deadline, and return shell exit code 124.
- Send canonical Jupyter `input_reply` messages, use `getpass` for password
  prompts, redact password values from history, and preserve the upstream
  contract that a custom stdin hook owns its reply.
- Reject invalid or conflicting accelerator flags before allocation instead of
  silently falling back to A100 or V6E1.
- Preserve allocation HTTP evidence while retaining the deprecated HTTP 412
  exception type for API compatibility; command output now gives concise
  usage/entitlement/capacity guidance instead of treating the class name as a
  confirmed diagnosis.
- Centralize the actual OAuth2 authentication default and route detached
  keep-alive children through the dependency-diagnostic entry point.
- Check the audited fork's GitHub releases and install exact fork tags; never
  replace this build with the package published under the upstream PyPI name.
- Retry only idempotent control-plane reads, reconcile lost assignment and
  unassignment POST responses without replaying them, and provide exact
  endpoint cleanup for an untracked server assignment.
- Use a live-validated 256 KiB default for resumable transfers, reconnect once
  for idempotent file controls, and reconcile an upload finalization response
  against the destination's size and SHA-256.
- Reject non-finite, non-positive, overflowing, and sub-byte transfer chunk
  sizes during CLI parsing, before session lookup or remote executor creation.
- Record orphan-assignment release history under a Windows-safe key and make
  the audit write best-effort after confirmed server-side release.

### Security

- Prevent password prompt values from entering structured history logs.

## [0.6.0] - 2026-06-16

### Changed

- **auth:** OAuth2 login now uses a remote copy-paste flow instead of a
  localhost callback server. The CLI prints an authorization URL with
  `redirect_uri=https://sdk.cloud.google.com/applicationdefaultauthcode.html`
  and `token_usage=remote`, then reads the pasted code from stdin. This works
  in headless/remote environments where a browser cannot reach a local
  callback port. (#54)

### Added

- **display output:** Rich rendering for `display_data` output via a shared
  `render_display_data()` helper. HTML is converted with `html2text` and
  rendered as Markdown, following a `text/markdown > text/html > text/plain`
  priority; `text/plain` is wrapped with `Text.from_ansi` to preserve embedded
  ANSI escapes. Applied consistently across `exec`, `console`/`repl`, and
  automation call sites. (#58)

### Fixed

- **keep-alive:** Replace the `RuntimeService/KeepAliveAssignment` RPC on
  `colab.pa.googleapis.com` with a Tunnel Frontend (TFE) HTTP ping
  (`GET /tun/m/<endpoint>/keep-alive/` with `X-Colab-Tunnel: Google`) on
  `colab.research.google.com`, authenticated by the user's own bearer token.
  The old RPC required `serviceusage` consumer access to Colab's internal
  project and returned HTTP 403 `USER_PROJECT_DENIED` for every external user,
  causing their sessions to be idle-pruned within minutes. The TFE ping needs
  no project entitlement; because the VM often does not answer on this path, a
  `ReadTimeout` is treated as success while genuine HTTP errors propagate.
  (#14, #61)

### Removed

- Dead grpc-web client-registry / API-key code path and the now-irrelevant
  `colaboratory`-scope / `pa.googleapis.com` pre-flight remediation messaging,
  superseded by the TFE keep-alive ping. (#61)

[0.6.0]: https://github.com/googlecolab/google-colab-cli/compare/v0.5.11...v0.6.0

[0.6.0.post1]: https://github.com/DGJK2301/google-colab-cli/compare/514db7e032a3e93bba9586cab8fcf00d37d1dd96...v0.6.0.post1

[0.7.0rc1]: https://github.com/DGJK2301/google-colab-cli/compare/v0.6.0.post1...v0.7.0rc1

[0.7.0rc2]: https://github.com/DGJK2301/google-colab-cli/compare/v0.7.0rc1...v0.7.0rc2

[0.7.0rc3]: https://github.com/DGJK2301/google-colab-cli/compare/v0.7.0rc2...v0.7.0rc3
