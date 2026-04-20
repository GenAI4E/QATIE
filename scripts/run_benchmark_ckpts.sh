#!/usr/bin/env bash
# ============================================================
# Benchmark all Lightning checkpoints under Source-Codes/ckpts:
# checkpoint size, param MiB, test PSNR/SSIM, forward latency (ms).
#
# Requires: run from anywhere; resolves paths relative to this script.
#
# Usage:
#   bash scripts/run_benchmark_ckpts.sh <data_dir> [extra args for benchmark_ckpts.py]
#
# <data_dir> — DPED dataset root (must contain iphone/… test patches).
#
# Examples:
#   cd submission/Source-Codes
#   bash scripts/run_benchmark_ckpts.sh /path/to/dped/dped
#   bash scripts/run_benchmark_ckpts.sh /path/to/dped/dped --device cpu --skip_timing
#
# Size + timing only (no dataset / no Lightning import for eval):
#   python benchmark_ckpts.py --skip_eval
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${ROOT_DIR}/ckpts"
OUT_CSV="${CKPT_DIR}/benchmark_results.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${1:?Usage: $0 <data_dir> [benchmark_ckpts.py args...]}"
shift

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m src.eval.benchmark_ckpts \
  --ckpt_dir "${CKPT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --output_csv "${OUT_CSV}" \
  "$@"
