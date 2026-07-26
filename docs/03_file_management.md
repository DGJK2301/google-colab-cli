---
log:
2026-07-25: Added Transfer Lifecycle v1: reusable Requests sessions, cross-process destination leases, process-identity-backed stale recovery, structured transfer JSON, and deterministic Ctrl+C resume evidence.
2026-07-21: Reduced the default transfer chunk to 256 KiB after a live free-CPU throughput probe showed the Colab tunnel could stall on 1 MiB request bodies. Upload body writes use a separate bounded timeout, and idempotent kernel-side file controls reconnect once after a stale transport. Finalization can reconcile a lost response by verifying an already-committed target against the requested size and SHA-256.
2026-07-20: Replaced whole-file base64 transfer with bounded, resumable Jupyter LargeFileManager chunks. Uploads and downloads now verify SHA-256 and size before atomic replacement; request timeouts are bounded and ambiguous upload responses are reconciled against remote size. The legacy Contents API remains in use for directory listing and deletion.
---

# Design: File Management (`ls`, `rm`, `upload`, `download`, `edit`)

## Overview

Directory operations use the Jupyter Contents API. File transfer adds a kernel
control channel for remote stat, hashing, fixed-size reads, and atomic commit.
This avoids loading an entire multi-megabyte file and its base64 expansion into
one HTTP request.

## Transfer Lease Contract

The CLI acquires one non-blocking local lease before it inspects a partial,
opens a Jupyter connection, or creates a remote executor. The lease remains
held through:

```text
partial inspection
→ prefix verification
→ chunk writes/reads
→ full SHA/size verification
→ atomic destination replacement
```

Upload identity is the SHA-256 of:

```text
runtime endpoint
normalized remote destination
```

The control-plane auth provider is deliberately excluded. Changing from OAuth
to ADC does not create a different file inside an already-recorded runtime and
must not permit a second writer. Local path spelling is preserved in output and
resume commands; platform case normalization is used only for lock identity.

Download identity is the canonical local destination. Two downloads from
different runtimes therefore cannot write the same local file concurrently.

Lease metadata is atomically written with restrictive permissions and records:

```text
PID + process start token
source/target/partial paths
source size/SHA-256
completed bytes and resume offset
retry count
heartbeat and lifecycle timestamps
```

If the OS lock is available but active metadata remains, it is reclaimed only
when the exact recorded process is proven gone or its PID has been reused.
Missing or unverifiable process identity fails closed.

## Upload Contract

`colab upload` defaults to 256 KiB source chunks and uses Jupyter Server's
`LargeFileManager` protocol:

1. Hash the local file without loading it into memory.
2. Select a deterministic temporary path from the destination and SHA-256.
3. If a partial file exists, compare the remote and local prefix hashes before
   resuming. A mismatched or oversized partial file is removed.
4. Send `chunk=1` to create/truncate, positive later chunks to append, and an
   empty `chunk=-1` marker to finalize Jupyter's save lifecycle.
5. Recompute remote size and SHA-256 in the kernel.
6. Atomically replace the destination with `os.replace` only after verification.

Every HTTP call has a connect/read timeout. If an upload response is lost, the
client queries the remote temporary size: the chunk is accepted only when the
size is exactly the before- or after-write boundary. Any other size fails.

Metadata and download calls keep the shorter control timeout. Upload calls use
a separate bounded budget because `requests` writes the JSON/base64 body before
waiting for the response. Kernel-side stat/read/remove/finalize controls reconnect
once after a transport failure; remote execution errors are not retried.

```bash
colab upload -s work --chunk-size-mib 0.25 --resume repo.bundle content/repo.bundle
```

`--no-resume` discards the deterministic partial file. `--no-overwrite`
prevents replacement of an existing final destination.

## HTTP Connection Lifecycle

A `ContentsClient` owns one `requests.Session`, so all upload chunks share the
same TLS and HTTP connection pool. Every request still supplies an explicit
`(connect_timeout, read_timeout)` tuple. Responses are closed after parsing,
and the Session is closed before the remote executor during cleanup.

The default chunk remains 256 KiB. `v0.7.0rc2` adds a benchmark-capable live
script for 0.25/0.5/1/2 MiB; it does not change the default before that evidence
is collected.

## Download Contract

`colab download` obtains the authoritative remote size/SHA-256, then reads
bounded base64 chunks through short kernel calls. Data is written to
`<target>.colab-download.part` with `fsync`; a verified partial download may be
resumed after prefix-hash comparison. The final local path is replaced only
after full size and SHA verification.

## Interruption and JSON Contract

`upload` and `download` support `--json`. Machine stdout is one
`colab.transfer.v1` document; progress and diagnostics remain on stderr.

Normal Ctrl+C:

```text
preserve the verified deterministic partial
→ record history state=interrupted
→ release local resources and lease
→ emit resume_argv/resume_command
→ exit 130
```

Cleanup failures are added to `warnings` and do not replace the original
transfer result or exception.

`retry_count` includes actual HTTP chunk replays and idempotent
control-channel reconnects. A lost response that is reconciled at the
already-committed byte boundary is not counted as a replay.

## Other Operations

- `ls`: `GET /api/contents/<path>`.
- `rm`: `DELETE /api/contents/<path>`.
- `edit`: verified download, local `$EDITOR`, then verified upload. Only an
  actual remote 404 creates an empty file; unrelated failures are not hidden.

## Scope

The transfer path is intended for source bundles, configuration, checkpoints,
and diagnostic artifacts. Multi-gigabyte datasets should stay in Drive, GCS,
or another object store and be localized from inside the VM. CLI transfer does
not turn the Jupyter control channel into a bulk data plane.

The chunk marker semantics follow Jupyter Server's
[`LargeFileManager`](https://github.com/jupyter-server/jupyter_server/blob/main/jupyter_server/services/contents/largefilemanager.py).

## Verification

Tests cover:

- bounded chunk markers and request timeouts;
- verified resume and mismatched-prefix reset;
- an ambiguous response after the server wrote a chunk;
- SHA/size verification before final replacement;
- interrupted upload and download resume with final SHA equality;
- fail-fast same-target concurrency and verified stale-lock recovery;
- JSON stdout purity, metrics, resume argv, and secret redaction;
- CLI cleanup on success and failure;
- an actual private-repository bundle round trip in the free CPU live test.
