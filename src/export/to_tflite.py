import os
import sys
from pathlib import Path
import argparse
import inspect
import onnx
import onnx2tf
import torch
import tensorflow as tf

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================== Model Builder ========================
from src.models.model_builder import build_model, load_checkpoint_weights, infer_channels_from_checkpoint

# ======================== Export Functions ========================


def _call_saved_model_default_signature(fn, input_tensor):
    """Call SavedModel ``serving_default`` with the correct keyword (e.g. ``input``)."""
    spec = fn.structured_input_signature
    if isinstance(spec, tuple) and len(spec) >= 2 and spec[1]:
        name = next(iter(spec[1].keys()))
        return fn(**{name: input_tensor})
    return fn(input_tensor)


def export_onnx(
    model,
    output_dir,
    onnx_filename,
    input_h=100,
    input_w=100,
    opset_version=18,
    legacy_onnx=False,
):
    """Export ONNX for the PyTorch→TF→TFLite pipeline.

    Default: opset 18 + dynamo exporter (no broken Resize down-convert to opset 11; onnx2tf-friendly).
    Use legacy_onnx=True with opset 11 for TorchScript export if you need legacy ONNX only.

    **Always exports static spatial dimensions** (``dynamic_axes=None``). Exporting with
    ``dynamic_axes`` and using ``overwrite_input_shape=...None,None`` in onnx2tf 1.28.x reliably
    hits ``node_Shape_*`` → ``convert_axis`` IndexError on U-Net-like graphs. onnx2tf must see a
    fixed ``1,3,input_h,input_w`` trace for conversion to succeed.

    ``input_h``/``input_w`` are **only** the trace resolution for ONNX→TF; they do **not** lock
    runtime inference to that size when you later build a TFLite with
    ``convert_tf_to_tflite(..., dynamic_shape=True)``, which wraps the SavedModel so I/O is
    ``[1,None,None,3]`` NHWC and the output is reshaped to match the input spatial size.
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, onnx_filename)
    dummy_input = torch.randn(1, 3, input_h, input_w)

    export_kwargs = dict(
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes=None,
    )
    sig = inspect.signature(torch.onnx.export)
    if "operator_export_type" in sig.parameters:
        export_kwargs["operator_export_type"] = torch.onnx.OperatorExportTypes.ONNX
    if legacy_onnx and "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    mode = f"legacy dynamo=False, opset {opset_version}" if legacy_onnx else f"dynamo exporter, opset {opset_version}"
    print(f"Exporting ONNX at {input_h}×{input_w}, {mode}...")
    torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)
    return onnx_path

def convert_onnx_to_tf(onnx_path, output_dir, input_h=100, input_w=100):
    """ONNX -> TF SavedModel via onnx2tf (always static ``1,3,H,W`` here).

    Do **not** use ``None`` for H/W: onnx2tf 1.28.x fails on ``Shape``/``convert_axis`` for this
    architecture (``node_Shape_*``, list index out of range). Flexible height/width in the final
    ``.tflite`` is done in ``convert_tf_to_tflite(..., dynamic_shape=True)``, not in onnx2tf.

    ``dynamic_shape`` is ignored for conversion flags; kept for API compatibility with callers.
    """
    overwrite_shape = [f"input:1,3,{input_h},{input_w}"]
    print(f"onnx2tf overwrite_input_shape: {overwrite_shape}")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        overwrite_input_shape=overwrite_shape,
        non_verbose=True,
    )


def _build_tflite_converter(saved_model_dir, dynamic_shape=False):
    if not dynamic_shape:
        return tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    model = tf.saved_model.load(saved_model_dir)
    raw_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_name = list(raw_func.structured_outputs.keys())[0]
    print(f"Dynamic TFLite wrapper output tensor: {output_name}")
 
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32, name="input")])
    def serving_fn(input_tensor):
        outputs = _call_saved_model_default_signature(raw_func, input_tensor)
        predictions = outputs[output_name]
        # The model is image-to-image, so the wrapped output should follow input HxW.
        return tf.reshape(predictions, tf.shape(input_tensor), name="output")

    concrete_func = serving_fn.get_concrete_function()
    return tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])


def convert_tf_to_tflite(saved_model_dir, tflite_path, dynamic_shape=False):
    """Convert SavedModel to FP32 TFLite, optionally keeping dynamic HxW."""
    print(f"Converting TF to TFLite. Dynamic mode: {dynamic_shape}")

    converter = _build_tflite_converter(saved_model_dir, dynamic_shape=dynamic_shape)

    # Configure converter for the mixed ONNX/TF graph emitted by onnx2tf.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model: {tflite_path}")


def convert_tf_to_tflite_int8(
    saved_model_dir,
    tflite_path,
    representative_dataset,
    dynamic_shape=False,
):
    """Convert SavedModel to full INT8 PTQ TFLite."""
    print(f"Converting TF SavedModel to INT8 PTQ TFLite. Dynamic mode: {dynamic_shape}")

    converter = _build_tflite_converter(saved_model_dir, dynamic_shape=dynamic_shape)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved full INT8 PTQ TFLite model: {tflite_path}")


def convert_pytorch_to_tflite(
    model: torch.nn.Module,
    output_dir,
    model_name: str,
    input_h: int,
    input_w: int,
    dynamic_shape: bool,
    opset_version: int = 18,
    legacy_onnx: bool = False,
) -> str:
    """Full pipeline: PyTorch model → ONNX → TF SavedModel → FP32 TFLite.

    ``input_h``/``input_w`` fix the traced graph for onnx2tf. If ``dynamic_shape`` is True,
    ``convert_tf_to_tflite`` emits a flexible-height/width TFLite whose output spatial size
    follows the input (same H×W), not the trace dimensions.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    onnx_path = export_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        input_h,
        input_w,
        opset_version=opset_version,
        legacy_onnx=legacy_onnx,
    )
    convert_onnx_to_tf(onnx_path, output_dir, input_h, input_w)
    tflite_filename = "model_none.tflite" if dynamic_shape else f"{model_name}.tflite"
    tflite_path = os.path.join(output_dir, tflite_filename)
    convert_tf_to_tflite(output_dir, tflite_path, dynamic_shape=dynamic_shape)
    return tflite_path


def convert_pytorch_to_int8_tflite(
    model: torch.nn.Module,
    output_dir,
    model_name: str,
    input_h: int,
    input_w: int,
    dynamic_shape: bool,
    representative_dataset,
    opset_version: int = 18,
    legacy_onnx: bool = False,
) -> str:
    """Full pipeline: PyTorch model → ONNX → TF SavedModel → INT8 PTQ TFLite."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    onnx_path = export_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        input_h,
        input_w,
        opset_version=opset_version,
        legacy_onnx=legacy_onnx,
    )
    convert_onnx_to_tf(onnx_path, output_dir, input_h, input_w)
    if dynamic_shape:
        tflite_filename = "model_none_int8_ptq.tflite"
    else:
        tflite_filename = f"{model_name}_int8_ptq.tflite"
    tflite_path = os.path.join(output_dir, tflite_filename)
    convert_tf_to_tflite_int8(
        output_dir,
        tflite_path,
        representative_dataset,
        dynamic_shape=dynamic_shape,
    )
    return tflite_path


if __name__ == "__main__":
    # Isolate TF/onnx2tf from other frameworks when running this script only.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--channels",
        type=int,
        default=0,
        help="Channel width (0 = infer from checkpoint)",
    )
    parser.add_argument("--input_h", type=int, default=100, help="ONNX/TF trace height (runtime flexible if --dynamic)")
    parser.add_argument("--input_w", type=int, default=100, help="ONNX/TF trace width")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="model_none.tflite with any H×W; output spatial size matches input",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset")
    parser.add_argument(
        "--legacy-onnx",
        dest="legacy_onnx",
        action="store_true",
        default=False,
        help="TorchScript ONNX (dynamo=False); often needs --opset 11",
    )

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]

    output_dir = os.path.join(args.output_dir, args.model_name)

    if args.channels and args.channels > 0:
        ch = args.channels
    else:
        try:
            ch = infer_channels_from_checkpoint(args.ckpt_path)
        except Exception:
            ch = 32

    model = build_model(ch)
    model = load_checkpoint_weights(model, args.ckpt_path)
    model.eval()

    tflite_path = convert_pytorch_to_tflite(
        model,
        output_dir,
        args.model_name,
        args.input_h,
        args.input_w,
        args.dynamic,
        opset_version=args.opset,
        legacy_onnx=args.legacy_onnx,
    )

    print(f"\n✅ SUCCESS! TFLite model: {tflite_path}")