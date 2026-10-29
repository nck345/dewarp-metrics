import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from python.ocr_eval import evaluate_ocr

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
                   output_csv=None, ocr_cache_json=None, ocr_engine='tesseract',
                   octave_bin='octave', num_workers=2, use_local_ssd=True,
                   skip_image=False, skip_ocr=False):
	repo_root = os.path.dirname(os.path.abspath(__file__))
	matlab_dir = os.path.join(repo_root, 'matlab')

	pred_dir = os.path.abspath(pred_dir)
	gt_dir = os.path.abspath(gt_dir)

	print('=' * 95)
	print('=== DEWARP-METRICS UNIVERSAL BENCHMARK EVALUATOR ===')
	print(f'Model Name       : {model_name}')
	print(f'Benchmark        : {benchmark}')
	print(f'Dewarped Dir     : {pred_dir}')
	print(f'Ground Truth Dir : {gt_dir}')
	print(f'Parallel Workers : {num_workers}')
	print(f'Image Metrics    : {"SKIP" if skip_image else f"GNU Octave / MATLAB (MS-SSIM-M & LD-M with Masking, {num_workers} workers)"}')
	print(f'Text Metrics     : {"SKIP" if skip_ocr else f"{ocr_engine.upper()} (CER & Levenshtein ED, {num_workers} workers)"}')
	print('=' * 95)

	work_pred_dir = pred_dir
	work_gt_dir = gt_dir
	if use_local_ssd and '/content/drive' in pred_dir:
		print('🚀 Syncing images to local NVMe SSD (/content/) for maximum I/O speed...')
		local_base = '/content/benchmark_cache'
		work_pred_dir = sync_to_local_ssd(pred_dir, f"{local_base}/pred_{model_name}_{benchmark}")
		work_gt_dir = sync_to_local_ssd(gt_dir, f"{local_base}/gt_{benchmark}")
		print('✅ Local SSD sync complete!')

	df_image = None
	df_ocr = None

	# 1. IMAGE METRICS EVALUATION (MS-SSIM-M & LD-M)
	if not skip_image:
		print(f'\n>>> [Step 1/2] Computing MS-SSIM-M and LD-M using GNU Octave ({num_workers} parallel workers)...')

		def run_octave_worker(part_idx):
			with tempfile.NamedTemporaryFile(suffix=f'_image_part{part_idx}.csv', delete=False) as tmp:
				part_csv = tmp.name

			oct_pred = work_pred_dir.replace('\\', '/')
			oct_gt = work_gt_dir.replace('\\', '/')
			oct_out = part_csv.replace('\\', '/')

			matlab_script = (
				f"addpath('{matlab_dir.replace('\\', '/')}'); "
				f"run_eval_octave('{oct_pred}', '{oct_gt}', '{oct_out}', '{benchmark}', {part_idx}, {num_workers}); "
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
				df_image = pd.concat(valid_dfs, ignore_index=True)
				df_image = df_image.sort_values(by=['Dewarped_Image']).reset_index(drop=True)
				df_image['Index'] = range(1, len(df_image) + 1)
		else:
			df_image = run_octave_worker(1)

	# 2. TEXT METRICS EVALUATION (CER & ED)
	if not skip_ocr:
		print(f'\n>>> [Step 2/2] Computing CER and Levenshtein ED using {ocr_engine.upper()}...')
		with tempfile.NamedTemporaryFile(suffix='_ocr_metrics.csv', delete=False) as tmp:
			ocr_csv = tmp.name

		evaluate_ocr(
			pred_dir=work_pred_dir,
			gt_dir=work_gt_dir,
			output_csv=ocr_csv,
			cache_json=ocr_cache_json,
			benchmark_name=benchmark,
			engine=ocr_engine,
			num_workers=num_workers
		)

		if os.path.exists(ocr_csv):
			df_ocr = pd.read_csv(ocr_csv)
			df_ocr = df_ocr[df_ocr['Index'] != 'SUMMARY'].copy()
			try:
				os.remove(ocr_csv)
			except Exception:
				pass

	# 3. MERGE METRICS INTO FINAL REPORT
	print('\n>>> Merging evaluation results...')
	if df_image is not None and not df_image.empty and df_ocr is not None and not df_ocr.empty:
		merged = pd.merge(df_image, df_ocr, on=['Dewarped_Image', 'Scan_GT'], how='inner', suffixes=('_img', '_ocr'))
		merged.insert(1, 'Model', model_name)
		merged['Index'] = range(1, len(merged) + 1)
		merged['Time_seconds'] = (pd.to_numeric(merged['Time_seconds_img'], errors='coerce') +
		                          pd.to_numeric(merged['Time_seconds_ocr'], errors='coerce')).round(2)
		final_df = merged[['Index', 'Model', 'Dewarped_Image', 'Scan_GT', 'MS_SSIM_M', 'LD_M', 'CER', 'ED', 'Time_seconds']].copy()
	elif df_image is not None and not df_image.empty:
		final_df = df_image.copy()
		final_df.insert(1, 'Model', model_name)
	elif df_ocr is not None and not df_ocr.empty:
		final_df = df_ocr.copy()
		final_df.insert(1, 'Model', model_name)
	else:
		print("❌ No evaluation data collected.")
		return

	# Add summary row
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
	parser.add_argument('--ocr_engine', type=str, default='tesseract', choices=['tesseract', 'rapidocr'])
	parser.add_argument('--octave_bin', type=str, default='octave', help='Path to GNU Octave executable')
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
		ocr_engine=args.ocr_engine,
		octave_bin=args.octave_bin,
		num_workers=args.workers,
		use_local_ssd=not args.no_ssd_cache,
		skip_image=args.skip_image,
		skip_ocr=args.skip_ocr
	)
