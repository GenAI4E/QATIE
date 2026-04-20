"""Run TFLite on one RGB image and save the output (same I/O convention as eval_tflite.py).

Supports arbitrary input resolutions via three strategies:
  1. Dynamic resize — resize the interpreter to the (padded) image size.
  2. Tiled inference — split into overlapping patches at the model's native size.
  3. Combined (default) — try dynamic resize first, fall back to tiling.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), None)
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_tflite import (
	_is_nchw_from_input_shape,
	_spatial_hw_from_input_shape,
	_is_dynamic_spatial_dim,
	load_tflite_model,
	resolve_eval_hw,
	run_tflite_inference,
	_resize_rgb_float_hw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single_tile(interpreter, input_details, output_details,
					 tile: np.ndarray, is_nchw: bool) -> np.ndarray:
	"""Run inference on one HWC float32 tile, return HWC float32 in [0, 1]."""
	if is_nchw:
		input_data = np.transpose(tile, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
	else:
		input_data = tile[np.newaxis, ...].astype(np.float32)
	output = run_tflite_inference(interpreter, input_details, output_details, input_data)
	output = output.squeeze(0)
	if output.ndim == 3 and output.shape[0] == 3:
		output = np.transpose(output, (1, 2, 0))
	return np.clip(output, 0.0, 1.0)


def _make_blend_mask(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
	"""Raised-cosine blending mask — 1.0 in centre, tapering at edges."""
	mask = np.ones((tile_h, tile_w, 1), dtype=np.float32)
	if overlap <= 0:
		return mask
	ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
	mask[:overlap, :, 0] *= ramp[:, None]
	mask[-overlap:, :, 0] *= ramp[::-1, None]
	mask[:, :overlap, 0] *= ramp[None, :]
	mask[:, -overlap:, 0] *= ramp[None, ::-1]
	return mask


def _resize_interpreter(interpreter, input_details, output_details,
						target_shape: list):
	"""Resize the interpreter to *target_shape* and return fresh details."""
	current = list(input_details[0]['shape'])
	if current != target_shape:
		interpreter.resize_tensor_input(input_details[0]['index'], target_shape)
		interpreter.allocate_tensors()
	return interpreter.get_input_details(), interpreter.get_output_details()


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def infer_tiled(interpreter, input_details, output_details,
				img: np.ndarray, is_nchw: bool,
				tile_h: int, tile_w: int,
				overlap: int = 16) -> np.ndarray:
	"""Process an arbitrary-size HWC image by running overlapping tiles."""
	h, w, _ = img.shape
	step_h = max(tile_h - overlap, 1)
	step_w = max(tile_w - overlap, 1)

	accumulator = np.zeros_like(img, dtype=np.float64)
	weight_map = np.zeros((h, w, 1), dtype=np.float64)
	mask = _make_blend_mask(tile_h, tile_w, overlap).astype(np.float64)

	y = 0
	while y < h:
		x = 0
		y2 = min(y + tile_h, h)
		y1 = y2 - tile_h
		if y1 < 0:
			y1 = 0
			y2 = min(tile_h, h)
		while x < w:
			x2 = min(x + tile_w, w)
			x1 = x2 - tile_w
			if x1 < 0:
				x1 = 0
				x2 = min(tile_w, w)

			tile = img[y1:y2, x1:x2, :]
			th_actual, tw_actual = tile.shape[0], tile.shape[1]

			if th_actual < tile_h or tw_actual < tile_w:
				padded = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
				padded[:th_actual, :tw_actual, :] = tile
				enhanced = _run_single_tile(
					interpreter, input_details, output_details, padded, is_nchw,
				)
				enhanced = enhanced[:th_actual, :tw_actual, :]
				m = mask[:th_actual, :tw_actual, :]
			else:
				enhanced = _run_single_tile(
					interpreter, input_details, output_details, tile, is_nchw,
				)
				m = mask

			accumulator[y1:y2, x1:x2, :] += enhanced * m
			weight_map[y1:y2, x1:x2, :] += m

			if x2 >= w:
				break
			x += step_w
		if y2 >= h:
			break
		y += step_h

	return (accumulator / np.maximum(weight_map, 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pad-and-crop inference (dynamic resize)
# ---------------------------------------------------------------------------

def infer_padded(interpreter, input_details, output_details,
				 img: np.ndarray, is_nchw: bool,
				 align: int = 32) -> np.ndarray:
	"""Pad the image to a multiple of *align*, resize the interpreter, run, crop back."""
	h, w, _ = img.shape
	pad_h = (align - h % align) % align
	pad_w = (align - w % align) % align
	padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
	ph, pw = padded.shape[0], padded.shape[1]

	target = [1, 3, ph, pw] if is_nchw else [1, ph, pw, 3]
	input_details, output_details = _resize_interpreter(
		interpreter, input_details, output_details, target,
	)

	enhanced = _run_single_tile(
		interpreter, input_details, output_details, padded, is_nchw,
	)
	return enhanced[:h, :w, :]


# ---------------------------------------------------------------------------
# Combined strategy: try dynamic resize, fall back to tiling
# ---------------------------------------------------------------------------

def infer_any_resolution(
	interpreter, input_details, output_details,
	img: np.ndarray, is_nchw: bool,
	align: int = 8,
	tile_overlap: int = 16,
) -> np.ndarray:
	"""Infer on an arbitrary-size image.

	First attempts pad-and-crop via ``resize_tensor_input``.  If that fails
	(e.g. XNNPACK, fixed-shape ops), falls back to tiled inference using the
	model's original spatial dimensions.
	"""
	input_shape = list(input_details[0]['shape'])
	h_graph, w_graph = _spatial_hw_from_input_shape(input_shape, is_nchw)
	h, w = img.shape[0], img.shape[1]

	graph_matches = (h_graph == h and w_graph == w)
	if graph_matches:
		return _run_single_tile(
			interpreter, input_details, output_details, img, is_nchw,
		)

	# --- attempt 1: dynamic resize (pad to alignment) ---
	try:
		result = infer_padded(
			interpreter, input_details, output_details,
			img, is_nchw, align=align,
		)
		return result
	except Exception as exc:
		print(f'  Dynamic resize failed ({exc}); falling back to tiled inference.')

	# --- attempt 2: tiled inference at model's native resolution ---
	if not _is_dynamic_spatial_dim(h_graph) and not _is_dynamic_spatial_dim(w_graph):
		orig_target = [1, 3, h_graph, w_graph] if is_nchw else [1, h_graph, w_graph, 3]
		try:
			input_details, output_details = _resize_interpreter(
				interpreter, input_details, output_details, orig_target,
			)
		except Exception:
			pass
		return infer_tiled(
			interpreter, input_details, output_details,
			img, is_nchw,
			tile_h=h_graph, tile_w=w_graph,
			overlap=tile_overlap,
		)

	raise RuntimeError(
		'Cannot infer: model has dynamic spatial dims and resize_tensor_input failed.'
	)


# ---------------------------------------------------------------------------
# Top-level single-image entry point
# ---------------------------------------------------------------------------

def infer_one_image(
	tflite_path: str,
	input_path: str,
	output_path: str,
	use_xnnpack: bool = False,
	eval_height: Optional[int] = None,
	eval_width: Optional[int] = None,
	strategy: str = 'auto',
	tile_overlap: int = 16,
	align: int = 32,
) -> None:
	img = np.array(Image.open(input_path).convert('RGB'), dtype=np.float32) / 255.0

	interpreter, input_details, output_details = load_tflite_model(
		tflite_path, use_xnnpack=use_xnnpack,
	)
	input_shape = list(input_details[0]['shape'])
	is_nchw = _is_nchw_from_input_shape(input_shape)

	h_img, w_img = img.shape[0], img.shape[1]

	if eval_height is not None and eval_width is not None:
		img = _resize_rgb_float_hw(img, eval_height, eval_width)
		h_img, w_img = eval_height, eval_width

	print(f'Image size: {h_img}×{w_img}  (NCHW layout: {is_nchw})')

	if strategy == 'resize':
		th, tw = resolve_eval_hw(
			input_details[0]['shape'], is_nchw,
			eval_height, eval_width, img,
		)
		target = [1, 3, th, tw] if is_nchw else [1, th, tw, 3]
		input_details, output_details = _resize_interpreter(
			interpreter, input_details, output_details, target,
		)
		phone = _resize_rgb_float_hw(img, th, tw)
		output = _run_single_tile(
			interpreter, input_details, output_details, phone, is_nchw,
		)
	elif strategy == 'tile':
		h_graph, w_graph = _spatial_hw_from_input_shape(input_shape, is_nchw)
		if _is_dynamic_spatial_dim(h_graph) or _is_dynamic_spatial_dim(w_graph):
			raise RuntimeError(
				'--strategy tile requires a model with fixed spatial dims, '
				f'but got graph shape {input_shape}.'
			)
		output = infer_tiled(
			interpreter, input_details, output_details,
			img, is_nchw,
			tile_h=h_graph, tile_w=w_graph,
			overlap=tile_overlap,
		)
	elif strategy == 'auto':
		output = infer_any_resolution(
			interpreter, input_details, output_details,
			img, is_nchw,
			align=align,
			tile_overlap=tile_overlap,
		)
	else:
		raise ValueError(f'Unknown strategy: {strategy!r}')

	output = np.clip(output, 0.0, 1.0)
	out_u8 = (output * 255.0).round().astype(np.uint8)
	os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
	Image.fromarray(out_u8, mode='RGB').save(output_path)
	print(f'Saved: {output_path}  ({out_u8.shape[1]}×{out_u8.shape[0]})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
	p = argparse.ArgumentParser(
		description='TFLite single-image inference with arbitrary-resolution support.',
	)
	p.add_argument('--tflite_file', type=str, required=True, help='Path to .tflite model')
	p.add_argument('--input', type=str, required=True, help='Input image path (RGB)')
	p.add_argument(
		'--output', type=str, default=None,
		help='Output image path (default: <input_stem>_tflite.png next to input)',
	)
	p.add_argument(
		'--use_xnnpack', action='store_true',
		help='Use XNNPACK (fixed shapes only; see eval_tflite.py).',
	)
	p.add_argument('--eval_height', type=int, default=None)
	p.add_argument('--eval_width', type=int, default=None)
	p.add_argument(
		'--strategy', type=str, default='auto',
		choices=['auto', 'resize', 'tile'],
		help='Inference strategy: "auto" tries dynamic resize then tiles, '
		'"resize" forces interpreter resize, "tile" forces tiled inference.',
	)
	p.add_argument(
		'--tile_overlap', type=int, default=20,
		help='Overlap in pixels between tiles (for --strategy tile/auto).',
	)
	p.add_argument(
		'--align', type=int, default=8,
		help='Pad image to multiples of this for dynamic resize (for --strategy auto).',
	)
	return p.parse_args()


def main():
	args = parse_args()
	if (args.eval_height is None) != (args.eval_width is None):
		raise SystemExit('Use both --eval_height and --eval_width, or neither.')

	out = args.output
	if not out:
		base, _ = os.path.splitext(args.input)
		out = f'{base}_tflite.png'

	infer_one_image(
		args.tflite_file,
		args.input,
		out,
		use_xnnpack=args.use_xnnpack,
		eval_height=args.eval_height,
		eval_width=args.eval_width,
		strategy=args.strategy,
		tile_overlap=args.tile_overlap,
		align=args.align,
	)


if __name__ == '__main__':
	main()
