# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The package version is derived from the git tag via `hatch-vcs`; each release
below corresponds to a tag of the same name.

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
