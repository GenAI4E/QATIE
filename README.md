# QATIE

Dat To-Thanh<sup>1,5</sup>, Nghia Nguyen-Trong<sup>2,5</sup>, Hoang Vo<sup>1,5</sup>, Hieu Bui-Minh<sup>3</sup>, Tinh-Anh Nguyen-Nhu<sup>4,5†</sup>

<sup>1</sup> University of Science, VNU-HCM, Vietnam  
<sup>2</sup> University of Information Technology, VNU-HCM, Vietnam  
<sup>3</sup> Da Nang University of Economics, Vietnam  
<sup>4</sup> Ho Chi Minh University of Technology, VNU-HCM, Vietnam  
<sup>5</sup> Vietnam National University, Ho Chi Minh City, Vietnam  
† Corresponding author

<p align="left">
  <a href="#"><img src="https://img.shields.io/badge/CVPRW-Paper-1f6feb?logo=ieee&logoColor=white" alt="CVPRW Paper"></a>
  <a href="https://arxiv.org/abs/2604.21743"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

This repository provides the official implementation of **Bridging the Training–Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement (QATIE)**. It features `Gated Encoding and Multi-Scale Refinement Network`, a lightweight architecture designed for image enhancement (IE), supported by Quantization-Aware Training (QAT) and an end-to-end export pipeline to TFLite. 


## ✨ Highlights
* **Hierarchical Gated Architecture**: hierarchical network architecture with gated encoder blocks and multiscale refinement to preserve fine-grained visual feature
* **Quantization-Aware Training (QAT)**: simulate the effects of low-precision representation during the training process, which allows the network to adapt and prevents the typical drop in quality seen with standard post-training quantization (PTQ).
* **Mobile-Ready Deployment Pipeline**: This framework supports a smooth export process to mobile inference frameworks like TensorFlow Lite, enabling both FP32 and highly optimized INT8 integer-only execution on standard commercial smartphones.
## 📰 News

- **[24/04/2026]** The official training, evaluation, and export code is published.
- **[26/03/2026]** 🎉 Our paper has been accepted at CVPRW-2026 🎉

## 🗂️ Repository Structure

For a detailed guide to the modules under `src/`, see `src/README.md`.

```text
.
├── README.md
├── requirements.txt
├── SKILL.md
├── ckpts/
│   ├── model_with_qat_c32.ckpt                     # Pytorch Lightning checkpoint
│   ├── model_with_qat_c32_int8_100x100.tflite      # TFLite checkpoint with 100x100 resolution
│   └── model_with_qat_c32_int8_1920x1080.tflite    # TFLite checkpoint with Full HD resolution
├── scripts/
│   ├── quantize_and_eval_qat_ckpt.sh    # Quantize/evaluate QAT checkpoints
│   ├── run_benchmark_ckpts.sh           # Benchmark checkpoint models
│   ├── run_ckpts_to_int8_tflite_eval.sh # Export/evaluate INT8 TFLite
│   └── run_ckpts_to_tflite_eval.sh      # Export/evaluate FP32 TFLite
└── src/
  ├── README.md                         # Detailed guide to src modules
  ├── checkpoint_tflite_utils.py        # Shared checkpoint export/eval helpers
  ├── ckpts_to_int8_tflite_eval.py      # Batch INT8 TFLite export + eval
  ├── ckpts_to_tflite_eval.py           # Batch FP32 TFLite export + eval
  ├── data/
  │   ├── data_aug.py                   # GPU augmentations (Kornia)
  │   └── dped_dataset.py               # DPED dataset + DataModule
  ├── eval/
  │   ├── benchmark_ckpts.py            # Benchmark checkpoint metrics/latency
  │   ├── benchmark_quantized.py        # Benchmark quantized artifacts
  │   ├── checkpoint_loading.py         # Shared checkpoint parsing/loading
  │   ├── eval_original_images.py       # Eval on original images
  │   ├── eval_pytorch.py               # PyTorch metric evaluation
  │   └── eval_tflite.py                # TFLite metric evaluation
  ├── export/
  │   ├── quantize.py                   # Convert checkpoints to INT8 (FX)
  │   └── to_tflite.py                  # Export pipeline to TFLite
  ├── infer/
  │   └── infer_tflite.py               # Single-image TFLite inference
  ├── models/
  │   └── model_builder.py              # QATIE model definition/builders
  └── train/
    ├── loss.py                       # Training losses
    ├── train_qat.py                  # Training entrypoint (FP/QAT)
    └── train_utils_builder.py        # Loss/scheduler builders
```

## Requirements

### Dataset
Download [DPED dataset](https://aiff22.github.io/#dataset) (patches for CNN training) and extract it into dped/ folder.

### Python
**Python < 3.12** (Required to prevent errors during the TFLite conversion process).

Install dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Optional)* For WandB logging, create a `.env` file in the root directory and add your `WANDB_API_KEY`.

## Training

You can train the model from scratch or use Quantization-Aware Training (QAT). The training script is fully integrated with PyTorch Lightning and Weights & Biases (WandB).

Example: QAT with BF16 precision and Cosine Warmup

```bash
python src/train/train_qat.py \
  --model_name model_c32_loss02_quantize_norestart \
  --channels 32 \
  --loss_version 2 \
  --precision bf16 \
  --qat True \
  --scheduler_type cosine_warmup \
  --warmup_epochs 5 \
  --warmup_start_factor 0.1 \
  --eta_min 5e-6 \
  --num_epochs 50 \
  --limit_train_batches 0.01 \
  --use_wandb False
```

## Export & Quantization

> Note (20/04/2026): current `onnx2tf` lib is currently undergone major changes so the TFLite conversion might not worked properly for the provided Pytorch checkpoints. Consider re-train or using our converted .tflite files

To deploy the model on mobile devices, export the trained PyTorch checkpoint to TFLite (FP32 or INT8).

Convert to TFLite (FP32)

```bash
python src/export/to_tflite.py \
  --ckpt_path /path/to/model.ckpt \
  --channels 24 \
  --output_dir /path/to/results \
  --model_name modelv7_c24
```

Convert to INT8 (PyTorch FX Graph)

```bash
python src/export/quantize.py \
  --ckpt_path /path/to/model.ckpt \
  --save_path /path/to/save/int8 \
  --data_dir /path/to/dped 
```

## Evaluation & Benchmarking

### Evaluate Metrics (PSNR / SSIM)

Evaluate a TFLite file:

```bash
python src/eval/eval_tflite.py \
  --tflite_file /path/to/results/model.tflite \
  --data_dir ./dataset/dped/dped \
  --output_csv /path/to/results/eval_one.csv
```

Evaluate all TFLite files in a directory:

```bash
python src/eval/eval_tflite.py \
  --tflite_dir /path/to/results \
  --data_dir ./dataset/dped/dped \
  --output_csv /path/to/results/eval_all.csv
```

### Run Benchmarks (Latency & Memory)

To measure the real-world efficiency of your checkpoints:

```bash
# Benchmark PyTorch Checkpoints
python src/eval/benchmark_ckpts.py \
  --ckpt_dir ./ckpts \
  --data_dir ./dataset/dped/dped \
  --output_csv ./ckpts/benchmark_results.csv

# Benchmark Quantized (INT8/TFLite) Models
python src/eval/benchmark_quantized.py \
  --model_dir ./results/int8 \
  --data_dir ./dataset/dped/dped
```

## Inference

Run inference on a single high-resolution image using TFLite (supports dynamic padding and tile-based overlapping):

```bash
python src/infer/infer_tflite.py \
  --tflite_file /path/to/model.tflite \
  --input /path/to/input.jpg \
  --strategy auto
```

## Citation
```

```

## License

This project is released under the MIT license. 
