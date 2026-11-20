import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm.auto import tqdm
from python.ocr_eval import evaluate_ocr, extract_doc_id, PAPER_SUBSET_IDS, PAPER_70_IDS
from python.ms_ssim_torch import HighPrecisionMSSSIM, evaluate_ms_ssim_pair

_drive_remount_lock = threading.Lock()

def remount_google_drive(mount_point='/content/drive', max_retries=3, delay=2.0):
	"""Attempts to remount Google Drive in Colab if transport endpoint is disconnected."""
	with _drive_remount_lock:
		my_drive = os.path.join(mount_point, 'MyDrive')
		if os.path.exists(my_drive):
			return True

		try:
			from google.colab import drive
			print(f"⚠️ Transport endpoint disconnected. Attempting to remount Google Drive at {mount_point}...")
			for attempt in range(1, max_retries + 1):
				try:
					subprocess.run(['fusermount', '-u', '-z', mount_point], capture_output=True)
					subprocess.run(['umount', '-l', mount_point], capture_output=True)
				except Exception:
					pass

				try:
					drive.mount(mount_point, force_remount=True)
					time.sleep(delay)
					if os.path.exists(my_drive):
						print("✅ Successfully remounted Google Drive!")
						return True
				except Exception as ex:
					print(f"   Remount attempt {attempt}/{max_retries} failed: {ex}")
					if attempt < max_retries:
						time.sleep(delay * (2 ** (attempt - 1)))
		except ImportError:
			pass
		except Exception as e:
			print(f"⚠️ Unexpected error while remounting drive: {e}")
		return False

def is_endpoint_error(err):
	"""Checks if an exception is Errno 107 (Transport endpoint is not connected)."""
	if isinstance(err, OSError):
		if getattr(err, 'errno', None) == 107:
			return True
	err_str = str(err).lower()
	return '107' in err_str or 'transport endpoint' in err_str or 'endpoint is not connected' in err_str

def copy_file_with_retry(s_path, d_path, max_retries=5, base_delay=0.5):
	"""Copies a single file with retry logic, exponential backoff, and drive remount on Errno 107."""
	for attempt in range(1, max_retries + 1):
		tmp_d_path = f"{d_path}.tmp_{os.getpid()}_{attempt}"
		try:
			if '/content/drive' in s_path and not os.path.exists('/content/drive/MyDrive'):
				remount_google_drive()

			src_size = os.path.getsize(s_path)
			if os.path.exists(d_path) and os.path.getsize(d_path) == src_size and src_size > 0:
				return True

			time.sleep(0.01)

			with open(s_path, 'rb') as fsrc, open(tmp_d_path, 'wb') as fdst:
				shutil.copyfileobj(fsrc, fdst, length=4 * 1024 * 1024)

			if os.path.exists(tmp_d_path) and os.path.getsize(tmp_d_path) == src_size:
				os.replace(tmp_d_path, d_path)
				return True
			else:
				if os.path.exists(tmp_d_path):
					try:
						os.remove(tmp_d_path)
					except Exception:
						pass
				raise OSError(f"Size mismatch after copying {os.path.basename(s_path)}")
		except OSError as e:
			if os.path.exists(tmp_d_path):
				try:
					os.remove(tmp_d_path)
				except Exception:
					pass

			if is_endpoint_error(e):
				print(f"⚠️ [Errno 107] Transport endpoint disconnected on {os.path.basename(s_path)} (attempt {attempt}/{max_retries}).")
				remount_google_drive()
			else:
				print(f"⚠️ OSError copying {os.path.basename(s_path)} (attempt {attempt}/{max_retries}): {e}")

			if attempt < max_retries:
				time.sleep(base_delay * (2 ** (attempt - 1)))
			else:
				return False
		except Exception as e:
			if os.path.exists(tmp_d_path):
				try:
					os.remove(tmp_d_path)
				except Exception:
					pass
			print(f"⚠️ Unexpected error copying {os.path.basename(s_path)}: {e}")
			return False
	return False

def sync_to_local_ssd(src_dir, dst_dir, max_retries=5):
	"""
	Syncs images from src_dir (e.g. Google Drive) to local SSD (dst_dir).
	Resilient to FUSE transport endpoint disconnects (Errno 107) with automatic
	remounting, exponential backoff, progress indication, and multi-pass retry.
	"""
	if not os.path.exists(src_dir):
		if '/content/drive' in src_dir:
			remount_google_drive()
		if not os.path.exists(src_dir):
			print(f"⚠️ Source directory does not exist: {src_dir}. Falling back to reading directly.")
			return src_dir

	try:
		os.makedirs(dst_dir, exist_ok=True)
	except Exception as e:
		print(f"⚠️ Could not create SSD directory {dst_dir}: {e}. Falling back to {src_dir}")
		return src_dir

	files = []
	for attempt in range(1, max_retries + 1):
		try:
			files = os.listdir(src_dir)
			break
		except OSError as e:
			if is_endpoint_error(e):
				print(f"⚠️ [Errno 107] Listing {src_dir} failed (attempt {attempt}/{max_retries}). Remounting drive...")
				remount_google_drive()
				time.sleep(1.0 * (2 ** (attempt - 1)))
			else:
				print(f"⚠️ Listing {src_dir} failed: {e}")
				break

	if not files:
		print(f"⚠️ No files found or failed to list {src_dir}. Falling back to {src_dir}")
		return src_dir

	img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
	img_files = [f for f in files if f.endswith(img_exts)]
	if not img_files:
		return src_dir

	to_copy = []
	for f in img_files:
		s_path = os.path.join(src_dir, f)
		d_path = os.path.join(dst_dir, f)
		try:
			if not (os.path.exists(d_path) and os.path.getsize(d_path) == os.path.getsize(s_path) and os.path.getsize(s_path) > 0):
				to_copy.append(f)
		except OSError:
			to_copy.append(f)

	if not to_copy:
		return dst_dir

	bm_name = os.path.basename(dst_dir)
	failed_files = []
	with tqdm(total=len(to_copy), desc=f"Syncing SSD [{bm_name}]", unit="img", ncols=95) as pbar:
		for f in to_copy:
			s_path = os.path.join(src_dir, f)
			d_path = os.path.join(dst_dir, f)
			if not copy_file_with_retry(s_path, d_path, max_retries=max_retries):
				failed_files.append(f)
			pbar.update(1)

	if failed_files:
		print(f"⚠️ {len(failed_files)}/{len(to_copy)} files failed initial sync. Remounting drive for second pass...")
		remount_google_drive()
		still_failed = []
		for f in failed_files:
			s_path = os.path.join(src_dir, f)
			d_path = os.path.join(dst_dir, f)
			if not copy_file_with_retry(s_path, d_path, max_retries=max_retries):
				still_failed.append(f)
		failed_files = still_failed

	if failed_files:
		print(f"⚠️ {len(failed_files)} files could not be synced to SSD. Direct drive reading will be used as fallback.")
		if len(failed_files) >= len(img_files):
			return src_dir

	return dst_dir

def safe_to_csv(df, path, max_retries=3, **kwargs):
	"""Writes DataFrame to CSV with retry logic and drive remount on transport disconnect."""
	for attempt in range(1, max_retries + 1):
		try:
			os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
			df.to_csv(path, **kwargs)
			return
		except OSError as e:
			if is_endpoint_error(e):
				print(f"⚠️ [Errno 107] Saving {path} failed (attempt {attempt}/{max_retries}). Remounting drive...")
				remount_google_drive()
				time.sleep(1.0 * (2 ** (attempt - 1)))
			else:
				if attempt == max_retries:
					raise
				time.sleep(1.0)

def ensure_octave_mex(repo_root, matlab_dir, octave_bin='octave'):
	"""Verifies SIFT Flow MEX binaries can be loaded and executed by Octave; auto-compiles if missing."""
	if shutil.which(octave_bin) is None:
		for cand in ['/usr/bin/octave', '/usr/local/bin/octave']:
			if os.path.exists(cand):
				octave_bin = cand
				break

	sift_dir = os.path.join(matlab_dir, 'SIFTflow')
	dense_dir = os.path.join(sift_dir, 'mexDenseSIFT')
	flow_dir = os.path.join(sift_dir, 'mexDiscreteFlow')

	def verify_runtime():
		test_script = (
			"try, pkg load image; catch, end; "
			f"addpath('{matlab_dir.replace('\\', '/')}', '-begin'); "
			f"addpath('{sift_dir.replace('\\', '/')}', '-begin'); "
			f"addpath('{dense_dir.replace('\\', '/')}', '-begin'); "
			f"addpath('{flow_dir.replace('\\', '/')}', '-begin'); "
			"try, "
			"  s = mexDenseSIFT(zeros(2,2,'double'), 3, 1); "
			"  f = imfilter(zeros(4,4,128,'uint8'), [1 2 1; 2 4 2; 1 2 1]/16, 'same', 'replicate'); "
			"  if (exist('mexDiscreteFlow') != 3 && exist('mexDiscreteFlow') != 103), exit(1); end; "
			"  exit(0); "
			"catch, "
			"  exit(1); "
			"end;"
		)
		try:
			res = subprocess.run([octave_bin, '--no-gui', '--silent', '--eval', test_script],
			                     capture_output=True, text=True, timeout=15)
			return res.returncode == 0
		except Exception:
			return False

	if verify_runtime():
		return True

	compile_sh = os.path.join(repo_root, 'compile_mex.sh')
	if os.path.exists(compile_sh):
		print("⚙️ SIFT Flow MEX binaries missing or uncallable in GNU Octave. Auto-compiling via compile_mex.sh...")
		try:
			comp_res = subprocess.run(['bash', compile_sh], cwd=repo_root, capture_output=True, text=True)
			if shutil.which(octave_bin) is None:
				for cand in ['/usr/bin/octave', '/usr/local/bin/octave']:
					if os.path.exists(cand):
						octave_bin = cand
						break
			if comp_res.returncode == 0 and verify_runtime():
				print("✅ Auto-compilation and verification of SIFT Flow MEX succeeded!")
				return True
			else:
				print("❌ Auto-compilation or verification failed:")
				if comp_res.stdout:
					print(comp_res.stdout.strip())
				if comp_res.stderr:
					print(comp_res.stderr.strip())
		except Exception as e:
			print(f"❌ Could not run compile_mex.sh: {e}")

	return False

def run_evaluation(pred_dir, gt_dir, model_name='Model', benchmark='docunet_paper_subset',
                   output_csv=None, ocr_cache_json=None, octave_bin='octave',
                   ms_ssim_backend='torch', num_workers=2, use_local_ssd=True,
                   skip_image=False, skip_ocr=False, save_details=False, **kwargs):
	repo_root = os.path.dirname(os.path.abspath(__file__))
	matlab_dir = os.path.join(repo_root, 'matlab')

	pred_dir = os.path.abspath(pred_dir)
	gt_dir = os.path.abspath(gt_dir)

	if shutil.which(octave_bin) is None:
		for cand in ['/usr/bin/octave', '/usr/local/bin/octave']:
			if os.path.exists(cand):
				octave_bin = cand
				break

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
		if work_pred_dir != pred_dir and work_gt_dir != gt_dir:
			print('✅ Local SSD sync complete!')
		else:
			print('⚠️ Local SSD sync encountered issues; fallback paths active.')

	img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
	try:
		all_pred = [f for f in os.listdir(work_pred_dir) if f.endswith(img_exts)]
	except OSError as e:
		if is_endpoint_error(e):
			remount_google_drive()
			all_pred = [f for f in os.listdir(pred_dir) if f.endswith(img_exts)]
		else:
			raise
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
			cand_local = os.path.join(work_gt_dir, f"{doc_id}{ext}")
			if os.path.exists(cand_local):
				scan_name = f"{doc_id}{ext}"
				scan_path = cand_local
				break
			cand_drive = os.path.join(gt_dir, f"{doc_id}{ext}")
			if os.path.exists(cand_drive):
				scan_name = f"{doc_id}{ext}"
				scan_path = cand_drive
				break

		if not scan_path:
			name_no_ext = os.path.splitext(p_name)[0]
			for ext in img_exts:
				cand_local = os.path.join(work_gt_dir, f"{name_no_ext}{ext}")
				if os.path.exists(cand_local):
					scan_name = f"{name_no_ext}{ext}"
					scan_path = cand_local
					break
				cand_drive = os.path.join(gt_dir, f"{name_no_ext}{ext}")
				if os.path.exists(cand_drive):
					scan_name = f"{name_no_ext}{ext}"
					scan_path = cand_drive
					break

		if scan_path:
			p_path = os.path.join(work_pred_dir, p_name)
			if not os.path.exists(p_path) and os.path.exists(os.path.join(pred_dir, p_name)):
				p_path = os.path.join(pred_dir, p_name)
			valid_pairs.append((p_name, p_path, scan_name, scan_path))

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
		with tqdm(total=len(valid_pairs), desc=f"MS-SSIM-M GPU [{benchmark}]", unit="img", ncols=95) as pbar:
			for p_name, p_path, s_name, s_path in valid_pairs:
				t0 = time.time()
				score = evaluate_ms_ssim_pair(p_path, s_path, model=model_ssim, device=dev)
				t_el = time.time() - t0
				ms_records.append({'Dewarped_Image': p_name, 'Scan_GT': s_name, 'MS_SSIM_M': score, 'Time_ssim': t_el})
				pbar.update(1)
				pbar.set_postfix({'Score': f"{score:.4f}"})
		df_ms_ssim = pd.DataFrame(ms_records)
		t_ssim_total = time.time() - t_ssim_start
		print(f'✅ MS-SSIM-M GPU Complete! Mean: {df_ms_ssim["MS_SSIM_M"].mean():.4f} in {t_ssim_total:.2f}s')

	# 2. LD-M EVALUATION (OCTAVE SIFT FLOW)
	if not skip_image:
		skip_octave_ms = (ms_ssim_backend == 'torch')
		step_name = '2/3' if ms_ssim_backend == 'torch' else '1/2'
		print(f'\n>>> [Step {step_name}] Computing LD-M via GNU Octave SIFT Flow ({num_workers} parallel workers)...')

		if not ensure_octave_mex(repo_root, matlab_dir, octave_bin):
			print("⚠️ CẢNH BÁO: Không thể tải hoặc biên dịch SIFT Flow MEX cho GNU Octave. Bỏ qua đo LD-M.")
		else:
			t_ldm_start = time.time()
			pbar_ldm = tqdm(total=len(valid_pairs), desc=f"LD-M Octave [{benchmark}]", unit="img", ncols=95)
			pbar_lock = threading.Lock()

			def run_octave_worker(part_idx):
				with tempfile.NamedTemporaryFile(suffix=f'_image_part{part_idx}.csv', delete=False) as tmp:
					part_csv = tmp.name

				oct_pred = work_pred_dir.replace('\\', '/')
				oct_gt = work_gt_dir.replace('\\', '/')
				oct_out = part_csv.replace('\\', '/')
				skip_ms_arg = 'true' if skip_octave_ms else 'false'

				matlab_script = (
					f"addpath('{matlab_dir.replace('\\', '/')}', '-begin'); "
					f"run_eval_octave('{oct_pred}', '{oct_gt}', '{oct_out}', '{benchmark}', {part_idx}, {num_workers}, {skip_ms_arg}); "
					f"exit;"
				)

				octave_cmd = [octave_bin, '--no-gui', '--silent', '--eval', matlab_script]
				err_lines = []
				try:
					proc = subprocess.Popen(
						octave_cmd,
						stdout=subprocess.PIPE,
						stderr=subprocess.STDOUT,
						text=True,
						bufsize=1
					)
					for line in proc.stdout:
						if 'PROGRESS_DONE:' in line:
							with pbar_lock:
								pbar_ldm.update(1)
						else:
							err_lines.append(line)
					proc.wait()
					if proc.returncode != 0:
						print(f"❌ Error in Octave worker {part_idx} (exit code {proc.returncode}):")
						for l in err_lines[-15:]:
							print(f"   {l.strip()}")
						if os.path.exists(part_csv):
							try:
								os.remove(part_csv)
							except Exception:
								pass
						return pd.DataFrame()
					if os.path.exists(part_csv):
						try:
							df = pd.read_csv(part_csv)
							df = df[df['Index'] != 'SUMMARY'].copy()
							os.remove(part_csv)
							return df
						except Exception as e:
							print(f"❌ Error reading Octave worker {part_idx} CSV: {e}")
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

			pbar_ldm.close()
			t_ldm_total = time.time() - t_ldm_start
			if df_ldm is not None and not df_ldm.empty:
				print(f'✅ LD-M Octave Complete! Mean: {df_ldm["LD_M"].mean():.2f} in {t_ldm_total:.2f}s')

	# 3. TEXT METRICS EVALUATION (TESSERACT 5.0)
	if not skip_ocr:
		step_name = '3/3' if ms_ssim_backend == 'torch' else '2/2'
		print(f'\n>>> [Step {step_name}] Computing CER and Levenshtein ED using Tesseract 5.0...')
		with tempfile.NamedTemporaryFile(suffix='_ocr_metrics.csv', delete=False) as tmp:
			ocr_csv = tmp.name

		ocr_summary, df_ocr = evaluate_ocr(
			pred_dir=work_pred_dir,
			gt_dir=work_gt_dir,
			benchmark_name=benchmark,
			output_csv=ocr_csv,
			cache_json=ocr_cache_json,
			num_workers=num_workers
		)
		if os.path.exists(ocr_csv):
			try:
				os.remove(ocr_csv)
			except Exception:
				pass

	# 4. MERGE AND SAVE RESULTS
	print('\n>>> Merging evaluation results...')
	base_df = pd.DataFrame({
		'Index': range(1, len(valid_pairs) + 1),
		'Model': model_name,
		'Benchmark': benchmark,
		'Dewarped_Image': [p[0] for p in valid_pairs],
		'Scan_GT': [p[2] for p in valid_pairs]
	})

	if df_ms_ssim is not None and not df_ms_ssim.empty:
		base_df = pd.merge(base_df, df_ms_ssim[['Dewarped_Image', 'Scan_GT', 'MS_SSIM_M', 'Time_ssim']], on=['Dewarped_Image', 'Scan_GT'], how='left')
	elif df_ldm is not None and not df_ldm.empty and 'MS_SSIM_M' in df_ldm.columns:
		base_df = pd.merge(base_df, df_ldm[['Dewarped_Image', 'Scan_GT', 'MS_SSIM_M']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	if df_ldm is not None and not df_ldm.empty:
		base_df = pd.merge(base_df, df_ldm[['Dewarped_Image', 'Scan_GT', 'LD_M', 'Time_seconds']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	if df_ocr is not None and not df_ocr.empty:
		base_df = pd.merge(base_df, df_ocr[['Dewarped_Image', 'Scan_GT', 'CER', 'ED']], on=['Dewarped_Image', 'Scan_GT'], how='left')

	# Summary row
	n_samples = len(valid_pairs)
	summary_dict = {'Index': 'SUMMARY', 'Model': model_name, 'Benchmark': benchmark, 'Images': n_samples}

	for col in ['MS_SSIM_M', 'LD_M', 'CER', 'ED']:
		if col in base_df.columns:
			summary_dict[col] = round(pd.to_numeric(base_df[col], errors='coerce').mean(), 4 if col in ['MS_SSIM_M', 'CER'] else 2)

	summary_df = pd.DataFrame([summary_dict])

	print('\n' + '=' * 85)
	print(f'=== FINAL BENCHMARK SUMMARY FOR [{model_name.upper()}] ON [{benchmark.upper()}] ===')
	for col, val in summary_dict.items():
		if col != 'Index':
			print(f'- {col:<15}: {val}')
	print('=' * 85)

	summary_only = summary_df.drop(columns=['Index'], errors='ignore')
	if output_csv:
		if save_details:
			detail_df = pd.concat([base_df, summary_df], ignore_index=True)
			safe_to_csv(detail_df, output_csv, index=False, encoding='utf-8')
			print(f'\nDetailed report saved to: {output_csv}')
		else:
			safe_to_csv(summary_only, output_csv, index=False, encoding='utf-8')
			print(f'\nSummary report saved to: {output_csv}')

	return summary_only

def run_benchmark_suite(base_dir, model_name='DocTr', benchmarks=None,
                        octave_bin='octave', ms_ssim_backend='torch',
                        num_workers=None, use_local_ssd=True,
                        skip_image=False, skip_ocr=False, save_details=True,
                        resume=True):
	if num_workers is None or num_workers < 1:
		num_workers = min(4, os.cpu_count() or 2)

	all_supported = ['docunet_paper_subset', 'docunet_130', 'dir300', 'docreal']
	if benchmarks is None or benchmarks == 'all' or (isinstance(benchmarks, (list, tuple)) and 'all' in benchmarks):
		target_bms = all_supported
	elif isinstance(benchmarks, str):
		target_bms = [benchmarks]
	else:
		target_bms = list(benchmarks)

	report_dir = os.path.join(base_dir, 'eval_reports')
	os.makedirs(report_dir, exist_ok=True)

	print('=' * 95)
	print(f'=== RUNNING BENCHMARK SUITE FOR MODEL [{model_name.upper()}] ===')
	print(f'Total Benchmarks : {len(target_bms)} ({", ".join(target_bms)})')
	print(f'Base Directory   : {base_dir}')
	print(f'Parallel Workers : {num_workers} workers')
	print(f'Resume mode      : {resume}')
	print('=' * 95)

	suite_summaries = []
	t_start = time.time()

	for idx, bm in enumerate(target_bms, 1):
		print('\n' + '#' * 95)
		print(f'# [{idx}/{len(target_bms)}] EVALUATING BENCHMARK: {bm.upper()}')
		print('#' * 95)

		bm_sum_csv = os.path.join(report_dir, f'{model_name}_{bm}_summary.csv')
		detail_csv = os.path.join(report_dir, f'{model_name}_{bm}_details.csv')

		if resume and os.path.exists(bm_sum_csv):
			try:
				df_cached = pd.read_csv(bm_sum_csv)
				if not df_cached.empty and 'Images' in df_cached.columns:
					is_complete = True
					if not skip_image:
						if 'MS_SSIM_M' not in df_cached.columns or pd.isna(df_cached['MS_SSIM_M'].iloc[0]):
							is_complete = False
						if 'LD_M' not in df_cached.columns or pd.isna(df_cached['LD_M'].iloc[0]):
							is_complete = False
					if not skip_ocr and ('CER' not in df_cached.columns or pd.isna(df_cached['CER'].iloc[0])):
						is_complete = False

					if is_complete:
						print(f'⚡ [RESUME] Đã tìm thấy báo cáo hoàn tất cho [{model_name.upper()}] trên [{bm.upper()}]. Bỏ qua đo lại.')
						suite_summaries.append(df_cached)
						continue
					else:
						print(f'🔄 [RESUME] Báo cáo trước đó của [{model_name.upper()}] trên [{bm.upper()}] bị thiếu chỉ số (LD-M hoặc OCR). Tiến hành đo lại...')
			except OSError as e:
				if is_endpoint_error(e):
					remount_google_drive()
					try:
						df_cached = pd.read_csv(bm_sum_csv)
					except Exception:
						df_cached = None
			except Exception:
				pass

		gt_dir = os.path.join(base_dir, 'benchmarks', bm, 'scan')
		if not os.path.exists(gt_dir) and bm in ['docunet_paper_subset', 'docunet_130']:
			gt_dir = os.path.join(base_dir, 'data_benchmark', 'scan', 'scan')

		pred_dir = os.path.join(base_dir, 'predictions', model_name, bm)
		if not os.path.exists(pred_dir) and bm in ['docunet_paper_subset', 'docunet_130']:
			pred_dir = os.path.join(base_dir, 'predictions', model_name)

		if not os.path.exists(gt_dir):
			print(f'Warning: Ground truth directory not found for {bm}: {gt_dir}. Skipping.')
			continue
		if not os.path.exists(pred_dir):
			print(f'Warning: Predictions directory not found for {bm}: {pred_dir}. Skipping.')
			continue

		cache_dir = os.path.join(base_dir, 'gt_cache')
		os.makedirs(cache_dir, exist_ok=True)
		ocr_cache_json = os.path.join(cache_dir, f'{bm}_tesseract_cache.json')

		target_output_csv = detail_csv if save_details else bm_sum_csv

		df_sum = run_evaluation(
			pred_dir=pred_dir,
			gt_dir=gt_dir,
			model_name=model_name,
			benchmark=bm,
			output_csv=target_output_csv,
			ocr_cache_json=ocr_cache_json,
			octave_bin=octave_bin,
			ms_ssim_backend=ms_ssim_backend,
			num_workers=num_workers,
			use_local_ssd=use_local_ssd,
			skip_image=skip_image,
			skip_ocr=skip_ocr,
			save_details=save_details
		)

		if df_sum is not None and not df_sum.empty:
			safe_to_csv(df_sum, bm_sum_csv, index=False, encoding='utf-8')
			suite_summaries.append(df_sum)

			# Incremental update of consolidated benchmark_summary.csv
			all_summary_csv = os.path.join(report_dir, 'benchmark_summary.csv')
			if os.path.exists(all_summary_csv):
				try:
					cur_master = pd.read_csv(all_summary_csv)
					cur_master = cur_master[~((cur_master['Model'] == model_name) & (cur_master['Benchmark'] == bm))]
					cur_master = pd.concat([cur_master, df_sum], ignore_index=True)
				except Exception:
					cur_master = df_sum
			else:
				cur_master = df_sum
			safe_to_csv(cur_master, all_summary_csv, index=False, encoding='utf-8')

	if suite_summaries:
		master_df = pd.concat(suite_summaries, ignore_index=True)
		report_dir = os.path.join(base_dir, 'eval_reports')
		os.makedirs(report_dir, exist_ok=True)

		# 1. Model summary CSV: <Model>_summary.csv
		model_summary_csv = os.path.join(report_dir, f'{model_name}_summary.csv')
		safe_to_csv(master_df, model_summary_csv, index=False, encoding='utf-8')

		# 2. Consolidated benchmark_summary.csv across all models
		all_summary_csv = os.path.join(report_dir, 'benchmark_summary.csv')
		if os.path.exists(all_summary_csv):
			try:
				existing_df = pd.read_csv(all_summary_csv)
				mask = (existing_df['Model'] == model_name) & (existing_df['Benchmark'].isin(master_df['Benchmark']))
				existing_df = existing_df[~mask]
				merged_df = pd.concat([existing_df, master_df], ignore_index=True)
			except Exception:
				merged_df = master_df
		else:
			merged_df = master_df
		safe_to_csv(merged_df, all_summary_csv, index=False, encoding='utf-8')

		print('\n' + '=' * 95)
		print(f'=== MASTER BENCHMARK SUMMARY FOR [{model_name.upper()}] ({time.time() - t_start:.1f}s) ===')
		print(master_df.to_string(index=False))
		print(f'\n✅ Model summary saved to           : {model_summary_csv}')
		print(f'✅ Consolidated leaderboard saved to: {all_summary_csv}')
		print('=' * 95)
		return master_df

	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dewarp-Metrics: Standard Benchmark Evaluation Pipeline')
	parser.add_argument('--base_dir', type=str, default=None, help='Base directory of Document_Dewarp_Benchmark')
	parser.add_argument('--benchmarks', nargs='+', default=None, help='List of benchmarks to evaluate, or all')
	parser.add_argument('--pred_dir', type=str, default=None, help='Directory of dewarped images')
	parser.add_argument('--gt_dir', type=str, default=None, help='Directory of ground truth scan images')
	parser.add_argument('--model_name', type=str, default='Model', help='Name of the evaluated model')
	parser.add_argument('--benchmark', type=str, default='docunet_paper_subset',
	                    choices=['docunet_paper_subset', 'docunet_paper_70', 'docunet_130', 'dir300', 'docreal', 'all'])
	parser.add_argument('--output_csv', type=str, default=None, help='Path to output merged CSV report')
	parser.add_argument('--ocr_cache_json', type=str, default=None, help='Path to cache GT OCR JSON')
	parser.add_argument('--octave_bin', type=str, default='octave', help='Path to GNU Octave executable')
	parser.add_argument('--ms_ssim_backend', type=str, default='torch', choices=['torch', 'octave'],
	                    help='Backend for MS-SSIM-M: torch (GPU float64 exact) or octave (CPU)')
	parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
	parser.add_argument('--no_ssd_cache', action='store_true', help='Disable local SSD caching')
	parser.add_argument('--skip_image', action='store_true', help='Skip image metrics (MS-SSIM-M & LD-M)')
	parser.add_argument('--skip_ocr', action='store_true', help='Skip OCR metrics (CER & ED)')
	parser.add_argument('--save_details', action='store_true', help='Save per-image detailed metrics CSV (default: False)')
	args = parser.parse_args()

	if args.base_dir:
		run_benchmark_suite(
			base_dir=args.base_dir,
			model_name=args.model_name,
			benchmarks=args.benchmarks if args.benchmarks else args.benchmark,
			octave_bin=args.octave_bin,
			ms_ssim_backend=args.ms_ssim_backend,
			num_workers=args.workers,
			use_local_ssd=not args.no_ssd_cache,
			skip_image=args.skip_image,
			skip_ocr=args.skip_ocr,
			save_details=args.save_details
		)
	else:
		if not args.pred_dir or not args.gt_dir:
			parser.error("Either --base_dir or both --pred_dir and --gt_dir must be provided.")
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
			skip_ocr=args.skip_ocr,
			save_details=args.save_details
		)
