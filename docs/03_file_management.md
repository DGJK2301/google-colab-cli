---
log:
2026-07-21: Reduced the default transfer chunk to 256 KiB after a live free-CPU throughput probe showed the Colab tunnel could stall on 1 MiB request bodies. Upload body writes use a separate bounded timeout, and idempotent kernel-side file controls reconnect once after a stale transport. Finalization can reconcile a lost response by verifying an already-committed target against the requested size and SHA-256.
2026-07-20: Replaced whole-file base64 transfer with bounded, resumable Jupyter LargeFileManager chunks. Uploads and downloads now verify SHA-256 and size before atomic replacement; request timeouts are bounded and ambiguous upload responses are reconciled against remote size. The legacy Contents API remains in use for directory listing and deletion.
---

# Design: File Management (`ls`, `rm`, `upload`, `download`, `edit`)

## Overview

Directory operations use the Jupyter Contents API. File transfer adds a kernel
control channel for remote stat, hashing, fixed-size reads, and atomic commit.
This avoids loading an entire multi-megabyte file and its base64 expansion into
one HTTP request.

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

## Download Contract

`colab download` obtains the authoritative remote size/SHA-256, then reads
bounded base64 chunks through short kernel calls. Data is written to
`<target>.colab-download.part` with `fsync`; a verified partial download may be
resumed after prefix-hash comparison. The final local path is replaced only
after full size and SHA verification.

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
- interrupted download resume;
- CLI cleanup on success and failure;
- an actual private-repository bundle round trip in the free CPU live test.
