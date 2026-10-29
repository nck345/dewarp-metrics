import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from python.ocr_eval import evaluate_ocr, extract_doc_id, PAPER_SUBSET_IDS, PAPER_70_IDS
from python.ms_ssim_torch import HighPrecisionMSSSIM, evaluate_ms_ssim_pair

def sync_to_local_ssd(src_dir, dst_dir):
	os.makedirs(dst_dir, exist_ok=True)
	files = os.listdir(src_dir)
	for f in files:
		s_path = os.path.join(src_dir, f)
		d_path = os.path.join(dst_dir, f)
		if os.path.isfile(s_path) and not os.path.exists(d_path):
			shutil.copy2(s_path, d_path)
	return dst_dir

def run_evaluation(pred_dir, gt_dir, model_name='Model', benchmark='docunet_paper_subset',
                   output_csv=None, ocr_cache_json=None, octave_bin='octave',
                   ms_ssim_backend='torch', num_workers=2, use_local_ssd=True,
                   skip_image=False, skip_ocr=False):
	repo_root = os.path.dirname(os.path.abspath(__file__))
	matlab_dir = os.path.join(repo_root, 'matlab')

	pred_dir = os.path.abspath(pred_dir)
	gt_dir = os.path.abspath(gt_dir)

	print('=' * 95)
	print('=== DEWARP-METRICS UNIVERSAL BENCHMARK EVALUATOR ===')
	print(f'Model Name        : {model_name}')
	print(f'Benchmark         : {benchmark}')
	print(f'Dewarped Dir      : {pred_dir}')
	print(f'Ground Truth Dir  : {gt_dir}')
	print(f'MS-SSIM-M Backend : {"GPU (PyTorch float64 - Exact)" if ms_ssim_backend == "torch" else "CPU (GNU Octave)"}')
	print(f'LD-M Backend      : GNU Octave SIFT Flow C++ MEX (-O3 + Scan SIFT Cache + {num_workers} Workers)')
	print(f'OCR Engine        : Tesseract 5.0 (GT Cache + {num_workers} Workers)')
	print('=' * 95)

	work_pred_dir = pred_dir
	work_gt_dir = gt_dir
	if use_local_ssd and '/content/drive' in pred_dir:
		print('🚀 Syncing images to local NVMe SSD (/content/) for maximum I/O speed...')
		local_base = '/content/benchmark_cache'
		work_pred_dir = sync_to_local_ssd(pred_dir, f"{local_base}/pred_{model_name}_{benchmark}")
		work_gt_dir = sync_to_local_ssd(gt_dir, f"{local_base}/gt_{benchmark}")
		print('✅ Local SSD sync complete!')

	# Build paired list
	img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
	all_pred = [f for f in os.listdir(work_pred_dir) if f.lower().endswith(img_exts)]
	all_pred.sort()

	valid_pairs = []
	for p_name in all_pred:
		doc_id = extract_doc_id(p_name)
		if benchmark == 'docunet_paper_subset':
			if doc_id not in PAPER_SUBSET_IDS:
				continue
		elif benchmark == 'docunet_paper_70':
			try:
				if int(doc_id) not in PAPER_70_IDS:
					continue
			except ValueError:
				continue

		scan_name = None
		scan_path = None
		for ext in img_exts:
			cand = os.path.join(work_gt_dir, f"{doc_id}{ext}")
			if os.path.exists(cand):
				scan_name = f"{doc_id}{ext}"
				scan_path = cand
				break
		if not scan_path:
			name_no_ext = os.path.splitext(p_name)[0]
			for ext in img_exts:
				cand = os.path.join(work_gt_dir, f"{name_no_ext}{ext}")
				if os.path.exists(cand):
					scan_name = f"{name_no_ext}{ext}"
					scan_path = cand
					break

		if scan_path:
			valid_pairs.append((p_name, os.path.join(work_pred_dir, p_name), scan_name, scan_path))

	df_ms_ssim = None
	df_ldm = None
	df_ocr = None

	# 1. MS-SSIM-M EVALUATION
	if not skip_image and ms_ssim_backend == 'torch':
		print('\n>>> [Step 1/3] Computing MS-SSIM-M on GPU via PyTorch (Double Precision float64)...')
		import torch
		dev = 'cuda' if torch.cuda.is_available() else 'cpu'
		model_ssim = HighPrecisionMSSSIM(device=dev)
		ms_records = []
		t_ssim_start = time.time()
		for idx, (p_name, p_path, s_name, s_path) in enumerate(valid_pairs, 1):
			t0 = time.time()
			score = evaluate_ms_ssim_pair(p_path, s_path, model=model_ssim, device=dev)
			t_el = time.time() - t0
			ms_records.append({'Dewarped_Image': p_name, 'Scan_GT': s_name, 'MS_SSIM_M': score, 'Time_ssim': t_el})
			print(f"[{idx}/{len(valid_pairs)}] {p_name:<25} -> {s_name:<10} | MS-SSIM-M: {score:.4f} ({t_el*1000:.1f}ms)")
		df_ms_ssim = pd.DataFrame(ms_records)
		t_ssim_total = time.time() - t_ssim_start
		print(f'✅ MS-SSIM-M GPU Complete! Mean: {df_ms_ssim["MS_SSIM_M"].mean():.4f} in {t_ssim_total:.2f}s')

	# 2. LD-M EVALUATION (OCTAVE SIFT FLOW)
	if not skip_image:
		skip_octave_ms = (ms_ssim_backend == 'torch')
		step_name = '2/3' if ms_ssim_backend == 'torch' else '1/2'
		print(f'\n>>> [Step {step_name}] Computing LD-M via GNU Octave SIFT Flow ({num_workers} parallel workers)...')

		def run_octave_worker(part_idx):
			with tempfile.NamedTemporaryFile(suffix=f'_image_part{part_idx}.csv', delete=False) as tmp:
				part_csv = tmp.name

			oct_pred = work_pred_dir.replace('\\', '/')
			oct_gt = work_gt_dir.replace('\\', '/')
			oct_out = part_csv.replace('\\', '/')
			skip_ms_arg = 'true' if skip_octave_ms else 'false'

			matlab_script = (
				f"addpath('{matlab_dir.replace('\\', '/')}'); "
				f"run_eval_octave('{oct_pred}', '{oct_gt}', '{oct_out}', '{benchmark}', {part_idx}, {num_workers}, {skip_ms_arg}); "
				f"exit;"
			)

			octave_cmd = [octave_bin, '--no-gui', '--silent', '--eval', matlab_script]
			try:
				subprocess.run(octave_cmd, check=True)
				if os.path.exists(part_csv):
					df = pd.read_csv(part_csv)
					df = df[df['Index'] != 'SUMMARY'].copy()
					try:
						os.remove(part_csv)
					except Exception:
						pass
					return df
			except Exception as e:
				print(f"❌ Error in Octave worker {part_idx}: {e}")
			return pd.DataFrame()

		if num_workers > 1:
			with ThreadPoolExecutor(max_workers=num_workers) as executor:
				part_dfs = list(executor.map(run_octave_worker, range(1, num_workers + 1)))
			valid_dfs = [df for df in part_dfs if not df.empty]
			if valid_dfs:
				df_ldm = pd.concat(valid_dfs, ignore_index=True)
				df_ldm = df_ldm.sort_values(by=['Dewarped_Image']).reset_index(drop=True)
		else:
			df_ldm = run_octave_worker(1)

	# 3. TEXT METRICS EVALUATION (TESSERACT 5.0)
	if not skip_ocr:
		step_name = '3/3' if ms_ssim_backend == 'torch' else '2/2'
		print(f'\n>>> [Step {step_name}] Computing CER and Levenshtein ED using Tesseract 5.0...')
		with tempfile.NamedTemporaryFile(suffix='_ocr_metrics.csv', delete=False) as tmp:
			ocr_csv = tmp.name

		evaluate_ocr(
			pred_dir=work_pred_dir,
			gt_dir=work_gt_dir,
			output_csv=ocr_csv,
			cache_json=ocr_cache_json,
			benchmark_name=benchmark,
			num_workers=num_workers
		)

		if os.path.exists(ocr_csv):
			df_ocr = pd.read_csv(ocr_csv)
			df_ocr = df_ocr[df_ocr['Index'] != 'SUMMARY'].copy()
			try:
				os.remove(ocr_csv)
			except Exception:
				pass

	# 4. MERGE METRICS INTO FINAL REPORT
	print('\n>>> Merging evaluation results...')
	base_df = pd.DataFrame({'Dewarped_Image': [p[0] for p in valid_pairs], 'Scan_GT': [p[2] for p in valid_pairs]})
	base_df.insert(0, 'Model', model_name)
	base_df.insert(0, 'Index', range(1, len(base_df) + 1))

	if df_ms_ssim is not None and not df_ms_ssim.empty:
		base_df = pd.merge(base_df, df_ms_ssim[['Dewarped_Image', 'Scan_GT', 'MS_SSIM_M']], on=['Dewarped_Image', 'Scan_GT'], how='left')
	elif df_ldm is not None and not df_ldm.empty and 'MS_SSIM_M' in df_ldm.columns:
		base_df = pd.merge(base_df, df_ldm[['Dewarped_Image', 'Scan_GT', 'MS_SSIM_M']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	if df_ldm is not None and not df_ldm.empty:
		base_df = pd.merge(base_df, df_ldm[['Dewarped_Image', 'Scan_GT', 'LD_M', 'Time_seconds']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	if df_ocr is not None and not df_ocr.empty:
		base_df = pd.merge(base_df, df_ocr[['Dewarped_Image', 'Scan_GT', 'CER', 'ED']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	final_df = base_df.copy()

	# Summary row
	n_samples = len(final_df)
	summary_dict = {'Index': 'SUMMARY', 'Model': model_name, 'Dewarped_Image': f'Total: {n_samples}', 'Scan_GT': 'All'}

	for col in ['MS_SSIM_M', 'LD_M', 'CER', 'ED', 'Time_seconds']:
		if col in final_df.columns:
			summary_dict[col] = round(pd.to_numeric(final_df[col], errors='coerce').mean(), 4 if col in ['MS_SSIM_M', 'CER'] else 2)

	summary_df = pd.DataFrame([summary_dict])
	report_df = pd.concat([final_df, summary_df], ignore_index=True)

	print('\n' + '=' * 95)
	print(f'=== FINAL BENCHMARK SUMMARY FOR [{model_name.upper()}] ON [{benchmark.upper()}] ===')
	for col, val in summary_dict.items():
		if col not in ['Index', 'Model', 'Dewarped_Image', 'Scan_GT']:
			print(f'- {col:<15}: {val}')
	print('=' * 95)

	if output_csv:
		os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
		report_df.to_csv(output_csv, index=False, encoding='utf-8')
		print(f'\n✅ Comprehensive report saved to: {output_csv}')

	return report_df

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dewarp-Metrics: Standard Benchmark Evaluation Pipeline')
	parser.add_argument('--pred_dir', type=str, required=True, help='Directory of dewarped images')
	parser.add_argument('--gt_dir', type=str, required=True, help='Directory of ground truth scan images')
	parser.add_argument('--model_name', type=str, default='Model', help='Name of the evaluated model')
	parser.add_argument('--benchmark', type=str, default='docunet_paper_subset',
	                    choices=['docunet_paper_subset', 'docunet_paper_70', 'docunet_130', 'all'])
	parser.add_argument('--output_csv', type=str, default=None, help='Path to output merged CSV report')
	parser.add_argument('--ocr_cache_json', type=str, default=None, help='Path to cache GT OCR JSON')
	parser.add_argument('--octave_bin', type=str, default='octave', help='Path to GNU Octave executable')
	parser.add_argument('--ms_ssim_backend', type=str, default='torch', choices=['torch', 'octave'],
	                    help='Backend for MS-SSIM-M: torch (GPU float64 exact) or octave (CPU)')
	parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
	parser.add_argument('--no_ssd_cache', action='store_true', help='Disable local SSD caching')
	parser.add_argument('--skip_image', action='store_true', help='Skip image metrics (MS-SSIM-M & LD-M)')
	parser.add_argument('--skip_ocr', action='store_true', help='Skip OCR metrics (CER & ED)')
	args = parser.parse_args()

	run_evaluation(
		pred_dir=args.pred_dir,
		gt_dir=args.gt_dir,
		model_name=args.model_name,
		benchmark=args.benchmark,
		output_csv=args.output_csv,
		ocr_cache_json=args.ocr_cache_json,
		octave_bin=args.octave_bin,
		ms_ssim_backend=args.ms_ssim_backend,
		num_workers=args.workers,
		use_local_ssd=not args.no_ssd_cache,
		skip_image=args.skip_image,
		skip_ocr=args.skip_ocr
	)
