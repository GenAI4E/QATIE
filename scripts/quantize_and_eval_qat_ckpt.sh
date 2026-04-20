#!/usr/bin/env bash
# Quantize both QAT and no-QAT checkpoints from a qat_ablation run (same as qat_ablation.sh).
# Outputs go to int8/with_qat/ and int8/no_qat/ so filenames do not overwrite.
#
# Usage:
#   ./quantize_and_eval_qat_ckpt.sh <data_dir> <results_dir> [channels] [loss_version] [no_qat_ckpt] [with_qat_ckpt] [legacy_qat_graph]
#
# legacy_qat_graph (7th arg, default 1): pass 1 to add --legacy_qat_graph (older QAT checkpoints
# trained before ConcatFP32 was kept in FP32). Pass 0 for checkpoints from current train_qat.py.
#
# If optional checkpoint paths are omitted, picks newest match under results_dir/checkpoints/ for:
#   qat_ablation_no_qat_c<channels>_loss<loss>_epoch*.ckpt
#   qat_ablation_with_qat_c<channels>_loss<loss>_epoch*.ckpt
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <data_dir> <results_dir> [channels] [loss_version] [no_qat_ckpt] [with_qat_ckpt] [legacy_qat_graph]}"
RESULTS_DIR="${2:?Usage: $0 <data_dir> <results_dir> [channels] [loss_version] [no_qat_ckpt] [with_qat_ckpt] [legacy_qat_graph]}"
CHANNELS="${3:-32}"
LOSS_VERSION="${4:-2}"
EXPLICIT_NO_QAT="${5:-}"
EXPLICIT_WITH_QAT="${6:-}"
# 1 = --legacy_qat_graph (default; matches pre–ConcatFP32-FP32 QAT checkpoints)
LEGACY_QAT_GRAPH="${7:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Override with: PYTHON_BIN=/path/to/python ./quantize_and_eval_qat_ckpt.sh ...
PYTHON_BIN="${PYTHON_BIN:-python}"
# Set RUN_BENCHMARK=0 to skip CSV benchmark (same metrics spirit as benchmark_ckpts.py)
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"

mkdir -p "${RESULTS_DIR}"
RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
INT8_DIR="${RESULTS_DIR}/int8"
mkdir -p "${CKPT_DIR}" "${INT8_DIR}"

RUN_NO_QAT="qat_ablation_no_qat_c${CHANNELS}_loss${LOSS_VERSION}"
RUN_WITH_QAT="qat_ablation_with_qat_c${CHANNELS}_loss${LOSS_VERSION}"

_abs_path() {
  local p="$1"
  [[ -n "$p" ]] || return 1
  [[ -f "$p" ]] || {
    echo "ERROR: not a file: $p"
    exit 1
  }
  echo "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
}

_pick_newest() {
  local run_prefix="$1"
  shopt -s nullglob
  local matches=("${CKPT_DIR}/${run_prefix}"_epoch*.ckpt)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo ""
    return 1
  fi
  ls -t "${matches[@]}" | sed -n '1p'
}

resolve_no_qat() {
  if [[ -n "${EXPLICIT_NO_QAT}" ]]; then
    _abs_path "${EXPLICIT_NO_QAT}"
  else
    local f
    f="$(_pick_newest "${RUN_NO_QAT}")" || true
    if [[ -z "${f}" ]]; then
      echo "ERROR: no checkpoint matching: ${CKPT_DIR}/${RUN_NO_QAT}_epoch*.ckpt"
      exit 1
    fi
    echo "${f}"
  fi
}

resolve_with_qat() {
  if [[ -n "${EXPLICIT_WITH_QAT}" ]]; then
    _abs_path "${EXPLICIT_WITH_QAT}"
  else
    local f
    f="$(_pick_newest "${RUN_WITH_QAT}")" || true
    if [[ -z "${f}" ]]; then
      echo "ERROR: no checkpoint matching: ${CKPT_DIR}/${RUN_WITH_QAT}_epoch*.ckpt"
      exit 1
    fi
    echo "${f}"
  fi
}

NO_QAT_CKPT="$(resolve_no_qat)"
WITH_QAT_CKPT="$(resolve_with_qat)"

SAVE_NO_QAT="${INT8_DIR}/no_qat"
SAVE_WITH_QAT="${INT8_DIR}/with_qat"
mkdir -p "${SAVE_NO_QAT}" "${SAVE_WITH_QAT}"

_run_quantize() {
  local label="$1"
  local ckpt="$2"
  local save_path="$3"
  local use_legacy="${4:-1}"
  echo "============================================================"
  echo "  ${label}"
  echo "============================================================"
  echo "  data_dir:   ${DATA_DIR}"
  echo "  save_path:  ${save_path}"
  echo "  channels:   ${CHANNELS}"
  echo "  checkpoint: ${ckpt}"
  echo "  legacy_qat_graph: ${use_legacy}"
  echo "  python:       ${PYTHON_BIN}"
  echo "============================================================"
  cd "${ROOT_DIR}"
  if [[ "${use_legacy}" == "1" || "${use_legacy}" == "true" || "${use_legacy}" == "yes" ]]; then
    "${PYTHON_BIN}" -m src.export.quantize \
      --ablation_model hybrid_base \
      --ckpt_path "${ckpt}" \
      --channels "${CHANNELS}" \
      --data_dir "${DATA_DIR}" \
      --save_path "${save_path}" \
      --legacy_qat_graph
  else
    "${PYTHON_BIN}" -m src.export.quantize \
      --ablation_model hybrid_base \
      --ckpt_path "${ckpt}" \
      --channels "${CHANNELS}" \
      --data_dir "${DATA_DIR}" \
      --save_path "${save_path}"
  fi
}

_run_quantize "Quantize with-QAT checkpoint → INT8" "${WITH_QAT_CKPT}" "${SAVE_WITH_QAT}" "${LEGACY_QAT_GRAPH}"
_run_quantize "Quantize no-QAT (FP32) checkpoint → INT8" "${NO_QAT_CKPT}" "${SAVE_NO_QAT}"

if [[ "${RUN_BENCHMARK}" == "1" || "${RUN_BENCHMARK}" == "true" || "${RUN_BENCHMARK}" == "yes" ]]; then
  echo "============================================================"
  echo "  benchmark_quantized.py (INT8 *_int8.pth under int8/)"
  echo "============================================================"
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" -m src.eval.benchmark_quantized \
    --model_dir "${INT8_DIR}" \
    --data_dir "${DATA_DIR}" \
    --device cpu \
    --output_csv "${INT8_DIR}/benchmark_quantized.csv"
fi

echo "Done."
echo "  no-QAT INT8:  ${SAVE_NO_QAT}"
echo "  with-QAT INT8: ${SAVE_WITH_QAT}"
if [[ "${RUN_BENCHMARK}" == "1" || "${RUN_BENCHMARK}" == "true" || "${RUN_BENCHMARK}" == "yes" ]]; then
  echo "  benchmark CSV: ${INT8_DIR}/benchmark_quantized.csv"
fi
