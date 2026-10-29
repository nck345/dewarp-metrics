import os
import sys
import argparse
import subprocess
import tempfile
import pandas as pd
from python.ocr_eval import evaluate_ocr

def run_evaluation(pred_dir, gt_dir, model_name='Model', benchmark='docunet_paper_subset',
                   output_csv=None, ocr_cache_json=None, ocr_engine='tesseract',
                   octave_bin='octave', skip_image=False, skip_ocr=False):
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
	print(f'Image Metrics    : {"SKIP" if skip_image else "GNU Octave / MATLAB (MS-SSIM-M & LD-M with Masking)"}')
	print(f'Text Metrics     : {"SKIP" if skip_ocr else f"{ocr_engine.upper()} (CER & Levenshtein ED)"}')
	print('=' * 95)

	df_image = None
	df_ocr = None

	# 1. IMAGE METRICS EVALUATION (MS-SSIM-M & LD-M)
	if not skip_image:
		print('\n>>> [Step 1/2] Computing MS-SSIM-M and LD-M using GNU Octave / MATLAB...')
		with tempfile.NamedTemporaryFile(suffix='_image_metrics.csv', delete=False) as tmp:
			octave_csv = tmp.name

		oct_pred = pred_dir.replace('\\', '/')
		oct_gt = gt_dir.replace('\\', '/')
		oct_out = octave_csv.replace('\\', '/')

		matlab_script = (
			f"addpath('{matlab_dir.replace('\\', '/')}'); "
			f"run_eval_octave('{oct_pred}', '{oct_gt}', '{oct_out}', '{benchmark}'); "
			f"exit;"
		)

		octave_cmd = [octave_bin, '--no-gui', '--silent', '--eval', matlab_script]
		try:
			subprocess.run(octave_cmd, check=True)
			if os.path.exists(octave_csv):
				df_image = pd.read_csv(octave_csv)
				df_image = df_image[df_image['Index'] != 'SUMMARY'].copy()
				try:
					os.remove(octave_csv)
				except Exception:
					pass
		except FileNotFoundError:
			print(f"❌ Error: '{octave_bin}' not found in PATH. Install octave via: apt-get install -y octave octave-image")
			if skip_ocr: return
		except subprocess.CalledProcessError as e:
			print(f"❌ Error during Octave execution (exit code {e.returncode}).")
			if skip_ocr: return

	# 2. TEXT METRICS EVALUATION (CER & ED)
	if not skip_ocr:
		print('\n>>> [Step 2/2] Computing CER and Levenshtein ED using OCR...')
		with tempfile.NamedTemporaryFile(suffix='_ocr_metrics.csv', delete=False) as tmp:
			ocr_csv = tmp.name

		evaluate_ocr(
			pred_dir=pred_dir,
			gt_dir=gt_dir,
			output_csv=ocr_csv,
			cache_json=ocr_cache_json,
			benchmark_name=benchmark,
			engine=ocr_engine
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
	if df_image is not None and df_ocr is not None:
		merged = pd.merge(df_image, df_ocr, on=['Dewarped_Image', 'Scan_GT'], how='inner', suffixes=('_img', '_ocr'))
		merged.insert(1, 'Model', model_name)
		merged['Index'] = range(1, len(merged) + 1)
		merged['Time_seconds'] = (pd.to_numeric(merged['Time_seconds_img'], errors='coerce') +
		                          pd.to_numeric(merged['Time_seconds_ocr'], errors='coerce')).round(2)
		final_df = merged[['Index', 'Model', 'Dewarped_Image', 'Scan_GT', 'MS_SSIM_M', 'LD_M', 'CER', 'ED', 'Time_seconds']].copy()
	elif df_image is not None:
		final_df = df_image.copy()
		final_df.insert(1, 'Model', model_name)
	elif df_ocr is not None:
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
		skip_image=args.skip_image,
		skip_ocr=args.skip_ocr
	)
