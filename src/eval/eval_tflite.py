import argparse
import csv
import os
import time
from glob import glob
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter


def _validate_lengths(ar, crop_width):
	ndim = ar.ndim
	try:
		crop_width = list(crop_width)
	except TypeError:
		return [(int(crop_width), int(crop_width))] * ndim
	if len(crop_width) == 1:
		crop_width = crop_width * ndim
	crops = []
	for c in crop_width:
		try:
			a, b = c
		except (TypeError, ValueError):
			a = b = c
		crops.append((int(a), int(b)))
	if len(crops) == 1:
		crops = crops * ndim
	return crops


def crop(ar, crop_width, copy=False, order='K'):
	ar = np.array(ar, copy=False)
	crops = _validate_lengths(ar, crop_width)
	slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
	if copy:
		return np.array(ar[slices], order=order, copy=True)
	return ar[slices]


def compare_ssim(x, y, win_size=None, data_range=None, gaussian_weights=False,
				 use_sample_covariance=True, **kwargs):
	if x.dtype != y.dtype:
		raise ValueError('Input images must have the same dtype.')
	if x.shape != y.shape:
		raise ValueError('Input images must have the same dimensions.')

	k1 = kwargs.pop('K1', 0.01)
	k2 = kwargs.pop('K2', 0.03)
	sigma = kwargs.pop('sigma', 1.5)

	if win_size is None:
		win_size = 11 if gaussian_weights else 7

	if data_range is None:
		data_range = 255

	if gaussian_weights:
		filter_func = gaussian_filter
		filter_args = {'sigma': sigma}
	else:
		filter_func = uniform_filter
		filter_args = {'size': win_size}

	x = x.astype(np.float64)
	y = y.astype(np.float64)

	ndim = x.ndim
	npix = win_size ** ndim
	cov_norm = npix / (npix - 1) if use_sample_covariance else 1.0

	ux = filter_func(x, **filter_args)
	uy = filter_func(y, **filter_args)
	uxx = filter_func(x * x, **filter_args)
	uyy = filter_func(y * y, **filter_args)
	uxy = filter_func(x * y, **filter_args)
	vx = cov_norm * (uxx - ux * ux)
	vy = cov_norm * (uyy - uy * uy)
	vxy = cov_norm * (uxy - ux * uy)

	c1 = (k1 * data_range) ** 2
	c2 = (k2 * data_range) ** 2

	a1 = 2 * ux * uy + c1
	a2 = 2 * vxy + c2
	b1 = ux ** 2 + uy ** 2 + c1
	b2 = vx + vy + c2
	s = (a1 * a2) / (b1 * b2)

	pad = (win_size - 1) // 2
	return crop(s, pad).mean()


def compute_psnr(pred, gt):
	mse = np.mean((pred - gt) ** 2)
	if mse < 1e-7:
		return 100.0
	return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, gt):
	pred_u8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
	gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)
	vals = [
		compare_ssim(gt_u8[:, :, c], pred_u8[:, :, c],
					 gaussian_weights=True, use_sample_covariance=False)
		for c in range(pred_u8.shape[2])
	]
	return float(np.mean(vals))


def _resize_rgb_float_hw(img: np.ndarray, height: int, width: int) -> np.ndarray:
	"""Resize HWC float32 image in [0, 1] to (height, width)."""
	if img.shape[0] == height and img.shape[1] == width:
		return img
	pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
	pil = pil.resize((width, height), Image.Resampling.BILINEAR)
	return np.asarray(pil, dtype=np.float32) / 255.0


def _dim_to_spatial_int(x) -> int:
	"""TFLite may use None or -1 for unknown spatial dims."""
	if x is None:
		return -1
	try:
		v = int(x)
	except (TypeError, ValueError):
		return -1
	return v


def _spatial_hw_from_input_shape(shape, is_nchw: bool):
	"""Return (H, W); unknown / dynamic dims become -1."""
	s = list(shape)
	if is_nchw:
		return _dim_to_spatial_int(s[2]), _dim_to_spatial_int(s[3])
	return _dim_to_spatial_int(s[1]), _dim_to_spatial_int(s[2])


def _is_dynamic_spatial_dim(d: int) -> bool:
	return d <= 0


def resolve_eval_hw(
	input_shape,
	is_nchw: bool,
	eval_height,
	eval_width,
	first_phone: np.ndarray,
):
	"""Pick one H×W for inference (single image or fixed batch resolution).

	If ``eval_height``/``eval_width`` are set, those win. Otherwise, if the model
	has fixed spatial dims in the graph, use them; if dynamic, use ``first_phone``'s size.
	"""
	if eval_height is not None and eval_width is not None:
		return int(eval_height), int(eval_width)
	h, w = _spatial_hw_from_input_shape(input_shape, is_nchw)
	if not _is_dynamic_spatial_dim(h) and not _is_dynamic_spatial_dim(w):
		return h, w
	ph, pw = int(first_phone.shape[0]), int(first_phone.shape[1])
	return ph, pw


def _is_nchw_from_input_shape(input_shape) -> bool:
	s = list(input_shape)
	return len(s) >= 4 and s[-1] != 3 and s[1] == 3


def load_tflite_model(tflite_path, use_xnnpack: bool = False):
	"""Load TFLite interpreter.

	Prefer LiteRT (``ai_edge_litert``): TensorFlow 2.20+ deprecates
	``tf.lite.Interpreter`` and that stack can segfault on some models or very
	large resolutions; LiteRT is the supported replacement.

	By default XNNPACK is disabled (``BUILTIN_WITHOUT_DEFAULT_DELEGATES``). The
	default CPU XNNPACK delegate often breaks on ``resize_tensor_input`` /
	``allocate_tensors`` for variable input sizes (reshape/node prepare errors).
	Enable ``use_xnnpack`` only if inputs are fixed-size and you want faster CPU
	inference.
	"""
	try:
		from ai_edge_litert.interpreter import Interpreter as _Interpreter
		from ai_edge_litert.interpreter import OpResolverType as _OpResolverType
	except ImportError:
		import tensorflow as tf

		_Interpreter = tf.lite.Interpreter
		_OpResolverType = tf.lite.experimental.OpResolverType

	resolver = _OpResolverType.AUTO if use_xnnpack else _OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
	interpreter = _Interpreter(
		model_path=tflite_path,
		experimental_op_resolver_type=resolver,
	)
	interpreter.allocate_tensors()
	return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def run_tflite_inference(interpreter, input_details, output_details, input_data):
	input_info = input_details[0]
	output_info = output_details[0]
	input_dtype = input_info['dtype']
	output_dtype = output_info['dtype']

	if np.issubdtype(input_dtype, np.integer):
		in_scale, in_zero = input_info.get('quantization', (0.0, 0))
		in_scale = float(in_scale)
		in_zero = int(in_zero)
		if in_scale <= 0:
			raise ValueError(f"Invalid input quantization scale: {in_scale}")
		q = np.round(input_data / in_scale + in_zero)
		qmin, qmax = np.iinfo(input_dtype).min, np.iinfo(input_dtype).max
		input_prepared = np.clip(q, qmin, qmax).astype(input_dtype)
	else:
		input_prepared = input_data.astype(input_dtype, copy=False)

	interpreter.set_tensor(input_info['index'], input_prepared)
	interpreter.invoke()
	raw_out = interpreter.get_tensor(output_info['index'])

	if np.issubdtype(output_dtype, np.integer):
		out_scale, out_zero = output_info.get('quantization', (0.0, 0))
		out_scale = float(out_scale)
		out_zero = int(out_zero)
		if out_scale <= 0:
			raise ValueError(f"Invalid output quantization scale: {out_scale}")
		return (raw_out.astype(np.float32) - out_zero) * out_scale
	return raw_out.astype(np.float32, copy=False)


def load_test_data(data_dir, full_hd: bool=False):
	phones = ['iphone', 'blackberry', 'sony']
	phone_paths = []
	canon_paths = []

	if not full_hd:
		for phone in phones:
			phone_dir = Path(data_dir) / phone / 'test_data' / 'patches' / phone
			canon_dir = Path(data_dir) / phone / 'test_data' / 'patches' / 'canon'
			if phone_dir.exists() and canon_dir.exists():
				p_files = sorted(glob(str(phone_dir / '*.jpg')))
				c_files = sorted(glob(str(canon_dir / '*.jpg')))
				if p_files and c_files:
					phone_paths.extend(p_files)
					canon_paths.extend(c_files)
		
		if not phone_paths:
			for phone in phones:
				phone_dir = Path(data_dir) / 'test_data' / 'patches' / phone
				canon_dir = Path(data_dir) / 'test_data' / 'patches' / 'canon'
				if phone_dir.exists() and canon_dir.exists():
					p_files = sorted(glob(str(phone_dir / '*.jpg')))
					c_files = sorted(glob(str(canon_dir / '*.jpg')))
					if p_files and c_files:
						phone_paths.extend(p_files)
						canon_paths.extend(c_files)
		if not phone_paths:
			raise ValueError(f'No phone images found under {data_dir}')
	else:
		for phone in phones:
			phone_dir = Path(data_dir) / phone
			if phone_dir.exists():
				p_files = sorted(glob(str(phone_dir / '*.jpg')))
				if p_files:
					phone_paths.extend(p_files)
		canon_dir = Path(data_dir) / 'canon'
		if canon_dir.exists():
			c_files = sorted(glob(str(canon_dir / '*.jpg')))
			canon_paths.extend(c_files)
		else:
			raise ValueError(f'No canon images found under {data_dir} or {canon_dir}')

	data = []
	sorted_phone_paths = sorted(phone_paths)
	sorted_canon_paths = sorted(canon_paths)
	if not full_hd:
		if len(sorted_phone_paths) != len(sorted_canon_paths):
			raise ValueError(f'Number of phone images ({len(sorted_phone_paths)}) does not match number of canon images ({len(sorted_canon_paths)})')
		for p_path, c_path in tqdm(zip(sorted_phone_paths, sorted_canon_paths), desc = "Loading test data"):
			phone_img = np.array(Image.open(p_path)).astype(np.float32) / 255.0
			canon_img = np.array(Image.open(c_path)).astype(np.float32) / 255.0
			data.append((phone_img, canon_img))
	else:
		# match based on filename of canon and phone images
		for c_path in tqdm(sorted_canon_paths, desc = "Loading canon images"):
			canon_img = np.array(Image.open(c_path)).astype(np.float32) / 255.0
			for p_path in sorted_phone_paths:
				if os.path.basename(c_path) == os.path.basename(p_path):
					phone_img = np.array(Image.open(p_path)).astype(np.float32) / 255.0
					data.append((phone_img, canon_img))
					# break
		if not data:
			raise ValueError(f'No matching phone images found for {canon_paths}')
	return data


def evaluate_tflite(
	tflite_path,
	test_data,
	use_xnnpack: bool = False,
	eval_height=None,
	eval_width=None,
):
	if (eval_height is None) != (eval_width is None):
		raise ValueError('Pass both eval_height and eval_width, or neither.')

	interpreter, input_details, output_details = load_tflite_model(
		tflite_path, use_xnnpack=use_xnnpack
	)
	input_shape = list(input_details[0]['shape'])
	is_nchw = _is_nchw_from_input_shape(input_shape)

	h_graph, w_graph = _spatial_hw_from_input_shape(input_shape, is_nchw)
	spatial_dynamic = _is_dynamic_spatial_dim(h_graph) or _is_dynamic_spatial_dim(w_graph)
	fixed_eval_hw = eval_height is not None and eval_width is not None

	if fixed_eval_hw:
		th, tw = int(eval_height), int(eval_width)
		print(f'  Eval resolution (forced): {th}×{tw}')
	elif not spatial_dynamic:
		th, tw = int(h_graph), int(w_graph)
		print(f'  Eval resolution (fixed graph): {th}×{tw}')
	else:
		th, tw = None, None
		print('  Spatially dynamic model: native H×W per image (output matches input size).')

	def _run_one(phone_img: np.ndarray, canon_img: np.ndarray) -> Tuple[float, float, float]:
		if is_nchw:
			input_data = np.transpose(phone_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
		else:
			input_data = phone_img[np.newaxis, ...].astype(np.float32)
		t0 = time.time()
		output = run_tflite_inference(interpreter, input_details, output_details, input_data)
		t_elapsed = time.time() - t0
		output = output.squeeze(0)
		if output.ndim == 3 and output.shape[0] == 3:
			output = np.transpose(output, (1, 2, 0))
		output = np.clip(output, 0, 1)
		if output.shape[0] != canon_img.shape[0] or output.shape[1] != canon_img.shape[1]:
			canon_img = _resize_rgb_float_hw(canon_img, output.shape[0], output.shape[1])
		return (
			compute_psnr(output, canon_img),
			compute_ssim(output, canon_img),
			t_elapsed,
		)

	psnr_vals = []
	ssim_vals = []
	times = []

	if th is not None and tw is not None:
		if is_nchw:
			target_shape = [1, 3, th, tw]
		else:
			target_shape = [1, th, tw, 3]
		if list(input_shape) != target_shape:
			interpreter.resize_tensor_input(input_details[0]['index'], target_shape)
			interpreter.allocate_tensors()
			input_details = interpreter.get_input_details()
			output_details = interpreter.get_output_details()

		for phone_img, canon_img in tqdm(test_data, desc="Evaluating TFLite model"):
			phone_img = _resize_rgb_float_hw(phone_img, th, tw)
			canon_img = _resize_rgb_float_hw(canon_img, th, tw)
			p, s, t_elapsed = _run_one(phone_img, canon_img)
			psnr_vals.append(p)
			ssim_vals.append(s)
			times.append(t_elapsed)
	else:
		prev_hw = None
		for phone_img, canon_img in tqdm(test_data, desc="Evaluating TFLite model"):
			th_i, tw_i = int(phone_img.shape[0]), int(phone_img.shape[1])
			if prev_hw != (th_i, tw_i):
				target_shape = [1, 3, th_i, tw_i] if is_nchw else [1, th_i, tw_i, 3]
				interpreter.resize_tensor_input(input_details[0]['index'], target_shape)
				interpreter.allocate_tensors()
				input_details = interpreter.get_input_details()
				output_details = interpreter.get_output_details()
				prev_hw = (th_i, tw_i)
			if canon_img.shape[0] != th_i or canon_img.shape[1] != tw_i:
				canon_img = _resize_rgb_float_hw(canon_img, th_i, tw_i)
			p, s, t_elapsed = _run_one(phone_img, canon_img)
			psnr_vals.append(p)
			ssim_vals.append(s)
			times.append(t_elapsed)

	return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), float(np.mean(times) * 1000)


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate model TFLite models on DPED test set')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--tflite_file', type=str, help='Single model .tflite file')
	group.add_argument('--tflite_dir', type=str, help='Directory containing model .tflite files (recursive)')

	parser.add_argument('--full_hd', action='store_true', help='Use full-hd test data')
	parser.add_argument('--data_dir', type=str, required=True, help='Path to DPED dataset root')
	parser.add_argument('--output_csv', type=str, default='eval_tflite_results.csv', help='Output CSV path')
	parser.add_argument(
		'--use_xnnpack',
		action='store_true',
		help='Use default TFLite delegates (XNNPACK). Faster on fixed input shapes; '
		'often fails with variable sizes (resize_tensor_input / allocate_tensors).',
	)
	parser.add_argument(
		'--eval_height',
		type=int,
		default=None,
		help='Resize all inputs to this height before inference. '
		'Use with --eval_width when the model has dynamic input or to force one resolution.',
	)
	parser.add_argument(
		'--eval_width',
		type=int,
		default=None,
		help='Resize all inputs to this width before inference (see --eval_height).',
	)
	return parser.parse_args()


def gather_models(args):
	if args.tflite_file:
		return [args.tflite_file]

	paths = sorted(glob(os.path.join(args.tflite_dir, '**', '*.tflite'), recursive=True))
	if not paths:
		raise ValueError(f'No .tflite files found under {args.tflite_dir}')
	return paths


def main():
	args = parse_args()
	if (args.eval_height is None) != (args.eval_width is None):
		raise SystemExit('Use both --eval_height and --eval_width, or neither.')

	test_data = load_test_data(args.data_dir, args.full_hd)
	if not test_data:
		raise ValueError(f'No DPED test image pairs found under {args.data_dir}')
	print(f'Found {len(test_data)} test image pair(s).')

	models = gather_models(args)
	print(f'Evaluating {len(models)} model TFLite file(s).')

	fieldnames = ['model_name', 'tflite_path', 'psnr_db', 'ssim', 'avg_inference_ms']
	os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

	with open(args.output_csv, 'w', newline='') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()

		for tflite_path in models:
			model_name = os.path.splitext(os.path.basename(tflite_path))[0]
			print(f'\nEvaluating: {model_name}')
			print(f'  Path: {tflite_path}')

			if not os.path.exists(tflite_path):
				print('  SKIP: file not found')
				continue

			psnr_db, ssim, avg_ms = evaluate_tflite(
				tflite_path,
				test_data,
				use_xnnpack=args.use_xnnpack,
				eval_height=args.eval_height,
				eval_width=args.eval_width,
			)
			print(f'  PSNR: {psnr_db:.4f} dB')
			print(f'  SSIM: {ssim:.4f}')
			print(f'  Avg inference: {avg_ms:.2f} ms')

			writer.writerow({
				'model_name': model_name,
				'tflite_path': tflite_path,
				'psnr_db': f'{psnr_db:.4f}',
				'ssim': f'{ssim:.4f}',
				'avg_inference_ms': f'{avg_ms:.2f}',
			})

	print(f'\nDone. Results saved to: {args.output_csv}')


if __name__ == '__main__':
	main()
