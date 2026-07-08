#!/bin/bash
#SBATCH --job-name=kempnerforge
#SBATCH --partition=<partition-name>     # e.g. gpu, kempner_h100
#SBATCH --account=<account-name>         # your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#
# Single-node multi-GPU training via torchrun.
#
# Usage:
#   sbatch scripts/slurm/singlenode.sh configs/train/7b.toml
#   sbatch scripts/slurm/singlenode.sh configs/train/7b.toml --train.max_steps=1000
#
# The first argument is the config TOML file. Any additional arguments are
# passed through as CLI overrides (e.g., --optimizer.lr=1e-4).

set -euo pipefail

# ---- Config ----
CONFIG="${1:?Usage: launch.sh <config.toml> [overrides...]}"
shift
OVERRIDES="$@"

NGPUS="${SLURM_GPUS_PER_NODE:-4}"

# ---- Network interface detection ----
# Detect the first UP InfiniBand interface with an IP address.
# Verified across all Kempner node types (H200, H100, A100):
#   - H200/H100: 4x CX-7 NDR (ib0-ib3), only ib0 has IP
#   - A100: 1x CX-6 HDR (ib0), has IP
# Without explicit IFNAME, gloo binds to em4 (management Ethernet) which
# can cause timeouts for multi-node collectives (wrong subnet / firewall).
IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"

# ---- Environment ----
# NCCL: use IB for RDMA data transport, IB interface for OOB bootstrap socket.
export NCCL_SOCKET_IFNAME="${IB_IFNAME}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Gloo: used by DCP async checkpoint for CPU-side coordination.
# Must bind to the same IB interface so peer connections route correctly.
export GLOO_SOCKET_IFNAME="${IB_IFNAME}"

# Ensure log directory exists
mkdir -p logs

echo "=== KempnerForge Training ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NGPUS}"
echo "IB iface:   ${IB_IFNAME}"
echo "Config:     ${CONFIG}"
echo "Overrides:  ${OVERRIDES}"
echo "============================"

# ---- Launch ----
# Background the trainer and forward SIGTERM so SLURM preemption / walltime actually
# reaches it. `--signal=B:SIGTERM@120` (above) signals THIS batch shell; a foreground
# torchrun would swallow that signal and the ranks would only die at the later SIGKILL
# — leaving no emergency checkpoint and the W&B run marked 'crashed'. `setsid` puts the
# trainer in its own process group so one `kill` hits uv + torchrun + every rank worker
# (robust to whether `uv run` itself relays signals). torchrun then forwards SIGTERM to
# the ranks, whose ShutdownHandler saves an emergency checkpoint and marks W&B 'preempted'.
setsid uv run torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    scripts/train.py "${CONFIG}" ${OVERRIDES} &
TRAIN_PID=$!

forward_term() {
    echo "[singlenode] SIGTERM received — forwarding to trainer group ${TRAIN_PID}"
    kill -TERM -"${TRAIN_PID}" 2>/dev/null || true
}
trap forward_term TERM INT

# `wait` returns >128 when interrupted by the trap, so loop until the child is truly
# gone. `|| rc=$?` keeps `set -e` from aborting on the signal-terminated exit; the final
# rc (143 on SIGTERM) is what the requeue chain reads to decide "preempted, resume".
rc=0
wait "${TRAIN_PID}" || rc=$?
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    wait "${TRAIN_PID}" || rc=$?
done
exit "${rc}"
