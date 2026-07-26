# Operational Recovery: doctor, monitor, and attach

## System model

```text
account assignment inventory
         ↓
local SessionState + keep-alive
         ↓
detached remote job
         ↓
foreground monitor
         ↓
durable local evidence directory
```

Colab VM lifetime and CLI process lifetime are independent. A detached job
can outlive a local terminal disconnect, but it cannot outlive VM
reclamation. Operational Recovery v1 therefore optimizes two things:

1. reconnect while the assignment still exists;
2. copy enough evidence locally that a later debugging session does not
   depend on a reclaimed `/content` filesystem.

## Doctor

```bash
colab doctor --json
colab doctor --json --network --timeout 10
```

The default command is entirely local. It reports package/version/install
identity, dependency versions, OAuth cache metadata and permissions,
strict session/settings parsing, keep-alive process state, and transfer
lease state. It reports only booleans for refresh and RAPT credentials.

`--network` performs one bounded assignment-list query. It does not assign,
connect, create a kernel, restart, release, or probe a runtime.

## Monitor

```bash
colab monitor train -s xoftr \
  --interval 5 \
  --probe-every 60 \
  --probe-timeout 20 \
  --output runs/train \
  --json
```

Files:

```text
stdout.log
stderr.log
job.jsonl
resources.jsonl
events.jsonl
monitor_state.json
summary.json
```

`monitor_state.json` atomically stores stdout/stderr offsets and binds the
evidence to job ID, session name, endpoint, remote runtime ID, and boot ID.
Re-running the command reconciles offsets against local file sizes and
continues without duplicating bytes.

One status poll supplies remote state, heartbeat, return code, error, log
sizes, runtime ID, and runner liveness. Tail calls are skipped when the
status size equals the local offset. When backlog exists, monitor drains up
to 64 chunks per poll with one local file open/fsync cycle. A terminal job
is not returned to the caller until all reported log bytes are local.

Resource probes are independent of job control. A probe timeout writes an
unavailable sample and monitoring continues. The probe reports actual GPU,
VRAM, utilization, temperature, host RAM, `/content` disk, and runtime boot
identity when available.

Ctrl+C exits 130, writes final state/summary, closes the control channel,
and never calls job cancellation.

## Attach

```bash
colab sessions --json
colab attach --endpoint <exact-endpoint> -s xoftr --json
```

Attach requires an exact endpoint from the current account assignment
list. It rejects local name and endpoint conflicts, verifies keep-alive,
writes recoverable local state, establishes a bounded control kernel by
default, starts local keep-alive, and publishes final state.

Local failure kills only a newly created local keep-alive process and
removes only the newly written local state. It never releases the remote
assignment. `--no-connect` defers control-kernel creation, but then the
first exec/jobs/monitor operation must establish it.

## Evidence limits

Monitor cannot recover bytes that were never copied before VM reclamation.
It also does not persist model state. Long training must write checkpoints
to Drive/GCS or another durable store. Local monitoring and durable
checkpoints solve different failure modes.
