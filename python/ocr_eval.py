import os
import json
import time
import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
	from python.checkpoint_utils import StepCkpt, is_endpoint_error, remount_google_drive, safe_json_dump
except ImportError:
	from checkpoint_utils import StepCkpt, is_endpoint_error, remount_google_drive, safe_json_dump

try:
	import pytesseract
except ImportError:
	try:
		import sys, subprocess
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytesseract"])
		import pytesseract
	except Exception:
		pytesseract = None

os.environ["OMP_THREAD_LIMIT"] = "1"

try:
	import rapidfuzz.distance.Levenshtein as rf_lev
	def levenshtein_dist(s1, s2):
		return rf_lev.distance(s1, s2)
except ImportError:
	try:
		import sys, subprocess
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"])
		import rapidfuzz.distance.Levenshtein as rf_lev
		def levenshtein_dist(s1, s2):
			return rf_lev.distance(s1, s2)
	except Exception:
		def levenshtein_dist(str1, str2):
			matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
			for i in range(1, len(str1) + 1):
				for j in range(1, len(str2) + 1):
					d = 0 if str1[i - 1] == str2[j - 1] else 1
					matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
			return matrix[len(str1)][len(str2)]

PAPER_70_IDS = {
	2, 5, 17, 19, 20, 23, 31, 37, 38, 39, 40, 41, 43, 45, 47, 48, 51, 54, 57, 60,
	61, 62, 64, 65, 67, 68, 70, 75, 76, 77, 78, 80, 81, 83, 84, 85, 87, 88, 90, 91,
	93, 96, 99, 100, 101, 102, 103, 104, 105, 134, 137, 138, 140, 150, 151, 155, 158,
	162, 163, 164, 165, 166, 169, 170, 172, 173, 175, 177, 178, 182
}

def extract_doc_id(filename):
	base = os.path.basename(filename)
	prefix = base.split('_')[0]
	digits = ''.join(filter(str.isdigit, prefix))
	return digits if digits else ''.join(filter(str.isdigit, base))

def run_tesseract_ocr(img_path, max_retries=3):
	global pytesseract
	if pytesseract is None:
		try:
			import pytesseract as pt
			pytesseract = pt
		except ImportError:
			try:
				import sys, subprocess
				subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytesseract"])
				import pytesseract as pt
				pytesseract = pt
			except Exception as err:
				print(f"Warning: pytesseract could not be imported/installed: {err}")
				return ''

	for attempt in range(1, max_retries + 1):
		try:
			return pytesseract.image_to_string(img_path)
		except Exception as e:
			if is_endpoint_error(e):
				remount_google_drive()
				time.sleep(1.0 * (2 ** (attempt - 1)))
			else:
				try:
					with Image.open(img_path) as img:
						return pytesseract.image_to_string(img)
				except Exception as inner_e:
					if attempt == max_retries:
						print(f"Warning: Tesseract failed on {img_path}: {inner_e}")
						return ''
					time.sleep(0.5)
	return ''

def evaluate_ocr(pred_dir, gt_dir, output_csv=None, cache_json=None, benchmark_name='docunet_130',
                 num_workers=2, ckpt_csv=None, resume=True, seed_df=None):
	gt_cache = {}
	if cache_json and os.path.exists(cache_json):
		try:
			with open(cache_json, 'r', encoding='utf-8') as f:
				gt_cache = json.load(f)
		except Exception:
			gt_cache = {}

	img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
	all_pred = []
	for attempt in range(1, 4):
		try:
			all_pred = [f for f in os.listdir(pred_dir) if f.lower().endswith(img_exts)]
			break
		except OSError as e:
			if is_endpoint_error(e):
				remount_google_drive()
				time.sleep(1.0 * (2 ** (attempt - 1)))
			else:
				raise
	all_pred.sort()

	valid_pairs = []
	for p_name in all_pred:
		doc_id = extract_doc_id(p_name)
		if benchmark_name == 'docunet_paper_70':
			try:
				if int(doc_id) not in PAPER_70_IDS:
					continue
			except ValueError:
				continue

		scan_name = None
		scan_path = None
		for ext in img_exts:
			cand = os.path.join(gt_dir, f"{doc_id}{ext}")
			if os.path.exists(cand):
				scan_name = f"{doc_id}{ext}"
				scan_path = cand
				break
		if not scan_path:
			name_no_ext = os.path.splitext(p_name)[0]
			for ext in img_exts:
				cand = os.path.join(gt_dir, f"{name_no_ext}{ext}")
				if os.path.exists(cand):
					scan_name = f"{name_no_ext}{ext}"
					scan_path = cand
					break

		if scan_path:
			valid_pairs.append((p_name, os.path.join(pred_dir, p_name), scan_name, scan_path))

	total = len(valid_pairs)
	print('=' * 85)
	print('=== Official Document Dewarping OCR Evaluation (CER & ED) ===')
	print(f'Benchmark        : {benchmark_name}')
	print(f'OCR Engine       : Tesseract 5.0 (Paper Standard)')
	print(f'Predictions dir  : {pred_dir}')
	print(f'Ground Truth dir : {gt_dir}')
	print(f'Total images     : {total}')
	print(f'Parallel workers : {num_workers}')
	print('=' * 85)

	if total == 0:
		print('No valid image pairs found.')
		return {}

	Ckpt = StepCkpt(ckpt_csv, ['Dewarped_Image', 'Scan_GT', 'CER', 'ED', 'Time_seconds']) if ckpt_csv else None
	Done_map = Ckpt.load(use_res=resume, seed_df=seed_df, val_cols=['CER', 'ED']) if Ckpt else {}

	indexed_items = list(enumerate(valid_pairs, 1))
	results = []
	pending_items = []
	for item in indexed_items:
		idx, (p_name, _, s_name, _) = item
		key = (p_name, s_name)
		if key in Done_map:
			rec = Done_map[key]
			cer_v = float(rec['CER'])
			ed_v = int(float(rec['ED']))
			el_v = float(rec['Time_seconds']) if rec.get('Time_seconds') not in (None, '') else 0.0
			results.append((idx, p_name, s_name, cer_v, ed_v, el_v))
		else:
			pending_items.append(item)

	if not pending_items:
		print(f'⚡ [RESUME] Đã khôi phục toàn bộ {total}/{total} ảnh OCR từ cache. Bỏ qua đo lại!')
	else:
		if results:
			print(f'⚡ [RESUME] Khôi phục {len(results)}/{total} ảnh OCR từ cache, tiếp tục đo {len(pending_items)} ảnh...')
		unique_scans = {}
		for _, (_, _, s_name, s_path) in pending_items:
			if s_name not in unique_scans:
				unique_scans[s_name] = s_path

		missing_scans = [s for s in unique_scans if s not in gt_cache]
		if missing_scans:
			gt_lock = threading.Lock()
			gt_dirty = False
			def scan_worker(s_name):
				return s_name, run_tesseract_ocr(unique_scans[s_name])

			try:
				with ThreadPoolExecutor(max_workers=min(num_workers, len(missing_scans))) as scan_exec:
					future_to_scan = {scan_exec.submit(scan_worker, s): s for s in missing_scans}
					with tqdm(total=len(missing_scans), desc=f"OCR Scan GT [{benchmark_name}]", unit="page", ncols=95) as pbar:
						done_cnt = 0
						for fut in as_completed(future_to_scan):
							s_name, text = fut.result()
							with gt_lock:
								gt_cache[s_name] = text
								gt_dirty = True
								done_cnt += 1
								snap = dict(gt_cache) if (cache_json and done_cnt % 10 == 0) else None
							if snap is not None:
								if safe_json_dump(snap, cache_json):
									gt_dirty = False
							pbar.update(1)
			finally:
				if cache_json and gt_dirty:
					with gt_lock:
						snap = dict(gt_cache)
					if safe_json_dump(snap, cache_json):
						print(f'Saved ground truth OCR cache to: {cache_json}')
		else:
			print(f'⚡ Loaded all {len(unique_scans)} ground truth scan texts from cache in 0.001s!')

		def process_dewarped_page(item):
			idx, (p_name, p_path, s_name, _) = item
			t_item = time.time()
			text_gt = gt_cache.get(s_name, '')
			text_pred = run_tesseract_ocr(p_path)
			ed = levenshtein_dist(text_gt, text_pred)
			cer = float(ed / max(1, len(text_gt)))
			elapsed = time.time() - t_item
			return (idx, p_name, s_name, cer, ed, elapsed)

		with tqdm(total=total, initial=len(results), desc=f"OCR Tesseract [{benchmark_name}]", unit="img", ncols=95) as pbar:
			if num_workers > 1:
				with ThreadPoolExecutor(max_workers=num_workers) as executor:
					future_map = {executor.submit(process_dewarped_page, item): item for item in pending_items}
					for fut in as_completed(future_map):
						res = fut.result()
						results.append(res)
						if Ckpt is not None:
							Ckpt.append({
								'Dewarped_Image': res[1],
								'Scan_GT': res[2],
								'CER': round(res[3], 6),
								'ED': int(res[4]),
								'Time_seconds': round(res[5], 3)
							})
						pbar.update(1)
						pbar.set_postfix({'CER': f"{np.mean([r[3] for r in results]):.4f}"})
			else:
				for item in pending_items:
					res = process_dewarped_page(item)
					results.append(res)
					if Ckpt is not None:
						Ckpt.append({
							'Dewarped_Image': res[1],
							'Scan_GT': res[2],
							'CER': round(res[3], 6),
							'ED': int(res[4]),
							'Time_seconds': round(res[5], 3)
						})
					pbar.update(1)
					pbar.set_postfix({'CER': f"{np.mean([r[3] for r in results]):.4f}"})

	cer_list, ed_list, time_list = [], [], []
	records = []
	t_start_all = time.time()

	results.sort(key=lambda x: x[0])
	for idx, p_name, s_name, cer, ed, elapsed in results:
		cer_list.append(cer)
		ed_list.append(ed)
		time_list.append(elapsed)
		records.append([idx, p_name, s_name, f"{cer:.4f}", ed, f"{elapsed:.2f}"])

	mean_cer = float(np.mean(cer_list))
	mean_ed = float(np.mean(ed_list))
	mean_time = float(np.mean(time_list))
	total_time = time.time() - t_start_all

	print('=' * 85)
	print('OVERALL TESSERACT OCR RESULTS:')
	print(f'- Total evaluated images : {total}')
	print(f'- Mean CER (↓)           : {mean_cer:.4f} ({mean_cer * 100:.2f}%)')
	print(f'- Mean Levenshtein ED (↓): {mean_ed:.2f} characters/page')
	print(f'- Mean time per image    : {mean_time:.2f} s')
	print(f'- Total elapsed time     : {total_time:.1f} s ({total_time / 60:.2f} min)')
	print('=' * 85)

	if output_csv:
		os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
		with open(output_csv, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['Index', 'Dewarped_Image', 'Scan_GT', 'CER', 'ED', 'Time_seconds'])
			writer.writerows(records)
			writer.writerow([])
			writer.writerow(['SUMMARY', f'Total: {total}', 'All', f'{mean_cer:.4f}', f'{mean_ed:.2f}', f'{mean_time:.2f}'])
		print(f'Saved Tesseract OCR report to CSV: {output_csv}')

	return {
		'count': total,
		'mean_cer': mean_cer,
		'mean_ed': mean_ed,
		'mean_time': mean_time,
		'total_time': total_time,
		'records': records
	}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Official Tesseract 5.0 OCR Evaluation for Document Dewarping')
	parser.add_argument('--pred_dir', type=str, required=True, help='Directory of dewarped images')
	parser.add_argument('--gt_dir', type=str, required=True, help='Directory of ground truth scan images')
	parser.add_argument('--output_csv', type=str, default=None, help='Path to output CSV report')
	parser.add_argument('--cache_json', type=str, default=None, help='Path to cache GT OCR JSON file')
	parser.add_argument('--benchmark', type=str, default='docunet_130',
	                    choices=['docunet_paper_70', 'docunet_130', 'dir300', 'docreal', 'all'])
	parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
	args = parser.parse_args()

	evaluate_ocr(
		pred_dir=args.pred_dir,
		gt_dir=args.gt_dir,
		output_csv=args.output_csv,
		cache_json=args.cache_json,
		benchmark_name=args.benchmark,
		num_workers=args.workers
	)
