# 📦 `src/` — Code Guide (Training → Export → Eval)

This folder contains the **core implementation**: dataset loading, model definition, training (FP32 + QAT), export to TFLite (FP32 + INT8 PTQ), and evaluation/benchmarking utilities.

## 🚀 Quick entrypoints

- **Train (FP32 / BF16 / QAT)**: `train/train_qat.py`
- **Export to TFLite (FP32)**: `export/to_tflite.py`
- **Convert Lightning checkpoint → INT8 FX model**: `export/quantize.py`
- **Evaluate**:
  - **PyTorch ckpt/model**: `eval/eval_pytorch.py`
  - **TFLite**: `eval/eval_tflite.py`
- **Benchmark**:
  - **Lightning checkpoints**: `eval/benchmark_ckpts.py`
  - **Quantized artifacts (INT8 .pth and optional .tflite)**: `eval/benchmark_quantized.py`
- **Batch export+eval for many checkpoints**:
  - **FP32 TFLite**: `ckpts_to_tflite_eval.py`
  - **INT8 PTQ TFLite**: `ckpts_to_int8_tflite_eval.py`

## 🧩 Top-level helpers

### `checkpoint_tflite_utils.py`
Shared helper used by multiple “for each checkpoint…” scripts.

- **What it does**
  - Loads a Lightning checkpoint safely and **extracts the model weights**
  - Resolves **channels** (from Lightning `hyper_parameters` or by probing weights)
  - Resolves **ablation variant** (from `hyper_parameters`, filename heuristics, or `--ablation_model`)
  - Skips fragile FX `GraphModule` pickles (the `torch.save(model)` outputs)
  - Provides small utilities for output paths and device row emission

- **Main API**
  - `load_checkpoint_model_for_export(ckpt_path, ablation_override=...)`
  - `resolve_output_paths(...)`
  - `load_eval_test_data_if_needed(...)`

## 🧱 Package layout

### 🗄️ `data/`

#### `data/dped_dataset.py`
DPED dataset loading and Lightning DataModule wiring.

- **Key responsibilities**
  - Reads DPED directory layouts (patch-based and full-resolution modes)
  - Provides train/val/test loaders used across training and eval scripts

#### `data/data_aug.py`
GPU-accelerated data augmentation (Kornia).

- **Used by**: training (`train/train_qat.py`) when augmentation is enabled

### 🧠 `models/`

#### `models/model_builder.py`
Single source of truth for:

- **Model definitions**
  - `HybridMixUNet` + ablation variants via `build_model(channels, model_name=...)`
- **Checkpoint utilities**
  - `load_checkpoint_weights(model, ckpt_path)` (Lightning → bare model)
  - `infer_channels_from_checkpoint(ckpt_path)` (auto-detect base channel width)
  - `is_fx_quantized_full_model_checkpoint(ckpt_path)` (detect fragile FX pickles)
  - `load_torch_checkpoint_trusted(ckpt_path)` (robust `torch.load` wrapper)
- **QAT support**
  - `build_qat_qconfig_mapping(...)` shared by training and INT8 conversion

### 🏋️ `train/`

#### `train/train_qat.py`
Main training entrypoint.

- **What it supports**
  - FP32 / BF16 training
  - FX Graph Mode QAT
  - W&B logging (optional)
  - Multiple scheduler and loss registry options

- **Typical usage**
  - Run this for both baseline training and QAT training; export scripts consume the produced Lightning `.ckpt`.

#### `train/loss.py`
Loss implementations used during training (PSNR-style terms, MS-SSIM, perceptual components, outlier-aware losses).

#### `train/train_utils_builder.py`
Registries/builders for losses and schedulers so `train_qat.py` stays configurable without duplicating logic.

### 📦 `export/`

#### `export/to_tflite.py`
End-to-end export pipeline:

1. **PyTorch → ONNX** (`export_onnx`)
2. **ONNX → TF SavedModel** (`convert_onnx_to_tf`, via `onnx2tf`)
3. **TF SavedModel → TFLite**
   - FP32: `convert_tf_to_tflite`
   - INT8 PTQ: `convert_tf_to_tflite_int8`

Notes:
- The ONNX→TF step uses a **fixed trace resolution** (H×W) for conversion stability.
- The optional `--dynamic` path wraps the SavedModel signature so the resulting TFLite can accept variable H×W.

#### `export/quantize.py`
Converts a Lightning checkpoint to an INT8 PyTorch FX model.

- **QAT checkpoints**: loads observer state from the checkpoint (no calibration needed)
- **FP32 checkpoints**: runs observer warmup (calibration) on DPED unless a pre-calibrated QAT state dict is provided

This is primarily for **PyTorch INT8 benchmarking** and for cases where you want a fully quantized FX model.

### 📈 `eval/`

#### `eval/checkpoint_loading.py`
Eval-focused helpers to keep checkpoint handling consistent across eval/benchmark scripts.

- `extract_stripped_model_state_dict(...)`
- `pick_ablation_and_load(...)`
- Hyperparameter + filename inference helpers

#### `eval/eval_pytorch.py`
Evaluates a PyTorch model on the DPED test split.

- Outputs **mean PSNR and SSIM** over the test set
- Used by benchmark scripts for repeated trials

#### `eval/eval_tflite.py`
Evaluates a `.tflite` model and reports:

- **PSNR / SSIM**
- **Average inference time (ms)** on the host CPU

Also contains:
- `load_tflite_model(...)` (LiteRT preferred, TF fallback)
- Quantize/dequantize helpers for INT8 models
- DPED-style test data loading helpers

#### `eval/benchmark_ckpts.py`
Benchmarks Lightning checkpoints under a directory:

- **Checkpoint file size**
- **Parameter memory** (MiB)
- **Mean/variance PSNR & SSIM** across repeated trials
- **Forward latency (ms)** on synthetic input

#### `eval/benchmark_quantized.py`
Benchmarks quantized artifacts under a directory:

- `*_int8.pth` (FX GraphModule pickles; can optionally **rebuild** from `.ckpt` if unpickling fails)
- Optional `*.tflite` (if `--include_tflite`)

Reports size, PSNR/SSIM statistics, and timing statistics similar to `benchmark_ckpts.py`.

#### `eval/eval_original_images.py`
Per-phone export + eval on `original_images/test` (matched by filename).

- Exports a **static** TFLite per phone resolution (config in `PHONE_CONFIGS`)
- Evaluates PSNR/SSIM against Canon ground-truth images
- Writes per-phone CSVs plus a combined CSV

### 🔎 `infer/`

#### `infer/infer_tflite.py`
Single-image inference using a TFLite model.

- Intended for quick manual checks and demo-like usage
- Supports strategies like auto padding / tile-based overlapping (depending on model)

## 🔁 Batch pipelines (scripts at `src/` root)

### `ckpts_to_tflite_eval.py`
For each checkpoint under a directory:

- Export to **FP32 TFLite**
- Evaluate PSNR/SSIM (and average inference ms)
- Write one CSV aggregating results

### `ckpts_to_int8_tflite_eval.py`
For each checkpoint under a directory:

- Export to **INT8 PTQ TFLite** using a DPED representative dataset
- Evaluate PSNR/SSIM
- Write one CSV aggregating results

