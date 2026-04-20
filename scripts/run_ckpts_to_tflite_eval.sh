#!/usr/bin/env bash
# ============================================================
# Export every Lightning checkpoint under Source-Codes/ckpts to TFLite
# (ONNX -> TF -> TFLite) and evaluate PSNR/SSIM with quantize.evaluate_model()
# on CPU and CUDA (see ckpts_to_tflite_eval.py).
#
# Usage:
#   bash scripts/run_ckpts_to_tflite_eval.sh <data_dir> [extra args for ckpts_to_tflite_eval.py]
#
# <data_dir> - DPED dataset root (must contain iphone/... test patches).
#
# Examples:
#   cd submission/Source-Codes
#   bash scripts/run_ckpts_to_tflite_eval.sh /path/to/dped/dped
#   bash scripts/run_ckpts_to_tflite_eval.sh /path/to/dped/dped --verbose --dynamic
#   bash scripts/run_ckpts_to_tflite_eval.sh /path/to/dped/dped --skip_cuda
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${ROOT_DIR}/ckpts"
OUT_CSV="${CKPT_DIR}/tflite_eval_cpu_cuda.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${1:?Usage: $0 <data_dir> [ckpts_to_tflite_eval.py args...]}"
shift

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m src.ckpts_to_tflite_eval \
  --ckpt_dir "${CKPT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --output_csv "${OUT_CSV}" \
  "$@"
