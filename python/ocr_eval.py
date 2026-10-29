import os
import json
import time
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

try:
	import rapidfuzz.distance.Levenshtein as rf_lev
	def levenshtein_dist(s1, s2):
		return rf_lev.distance(s1, s2)
except ImportError:
	def levenshtein_dist(str1, str2):
		matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
		for i in range(1, len(str1) + 1):
			for j in range(1, len(str2) + 1):
				d = 0 if str1[i - 1] == str2[j - 1] else 1
				matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
		return matrix[len(str1)][len(str2)]

PAPER_SUBSET_IDS = {
	'1', '2', '3', '4', '5', '6', '7', '9', '10', '21',
	'22', '23', '24', '27', '30', '31', '32', '36', '38', '40',
	'41', '44', '45', '46', '47', '48', '50', '51', '52', '53'
}

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

def get_ocr_engine(engine_name='tesseract'):
	if engine_name.lower() == 'tesseract':
		import pytesseract
		def ocr_func(img_path):
			try:
				img = Image.open(img_path)
				return pytesseract.image_to_string(img)
			except Exception:
				return ''
		return ocr_func, 'Tesseract 5.0 (CPU - Paper Standard)'
	elif engine_name.lower() == 'rapidocr':
		import cv2
		from rapidocr_onnxruntime import RapidOCR
		ocr = RapidOCR(det_use_cuda=True, rec_use_cuda=True, rec_batch_num=256, use_angle_cls=False)
		def ocr_func(img_path):
			bgr = cv2.imread(img_path)
			if bgr is None: return ''
			rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			res, _ = ocr(rgb)
			return ' '.join([line[1] for line in res]) if res else ''
		return ocr_func, 'RapidOCR (GPU Accelerated)'
	else:
		raise ValueError(f"Unsupported OCR engine: {engine_name}. Choose 'tesseract' or 'rapidocr'.")

def evaluate_ocr(pred_dir, gt_dir, output_csv=None, cache_json=None, benchmark_name='docunet_paper_subset',
                 engine='tesseract', num_workers=2):
	ocr_func, engine_desc = get_ocr_engine(engine)
	gt_cache = {}
	if cache_json and os.path.exists(cache_json):
		try:
			with open(cache_json, 'r', encoding='utf-8') as f:
				gt_cache = json.load(f)
		except Exception:
			gt_cache = {}

	img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
	all_pred = [f for f in os.listdir(pred_dir) if f.lower().endswith(img_exts)]
	all_pred.sort()

	valid_pairs = []
	for p_name in all_pred:
		doc_id = extract_doc_id(p_name)
		if benchmark_name == 'docunet_paper_subset':
			if doc_id not in PAPER_SUBSET_IDS:
				continue
		elif benchmark_name == 'docunet_paper_70':
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
	print(f'OCR Engine       : {engine_desc}')
	print(f'Predictions dir  : {pred_dir}')
	print(f'Ground Truth dir : {gt_dir}')
	print(f'Total images     : {total}')
	print(f'Parallel workers : {num_workers if engine == "tesseract" else 1}')
	print('=' * 85)

	if total == 0:
		print('No valid image pairs found.')
		return {}

	# Precompute missing ground truth scan cache
	unique_scans = {}
	for _, _, s_name, s_path in valid_pairs:
		if s_name not in unique_scans:
			unique_scans[s_name] = s_path

	missing_scans = [s for s in unique_scans if s not in gt_cache]
	if missing_scans:
		print(f'Scanning {len(missing_scans)} missing ground truth scan pages...')
		for s_name in missing_scans:
			gt_cache[s_name] = ocr_func(unique_scans[s_name])
		if cache_json:
			try:
				os.makedirs(os.path.dirname(os.path.abspath(cache_json)), exist_ok=True)
				with open(cache_json, 'w', encoding='utf-8') as f:
					json.dump(gt_cache, f, ensure_ascii=False, indent=2)
			except Exception:
				pass
	else:
		print(f'⚡ Loaded all {len(unique_scans)} ground truth scan texts from cache!')

	print(f"{'STT':<5} | {'Dewarped Image':<25} | {'Scan GT':<12} | {'CER':<8} | {'ED':<7} | {'Time (s)':<8}")
	print('-' * 85)

	cer_list, ed_list, time_list = [], [], []
	records = []
	t_start_all = time.time()

	def process_item(item):
		idx, (p_name, p_path, s_name, _) = item
		t_item = time.time()
		text_gt = gt_cache.get(s_name, '')
		text_pred = ocr_func(p_path)
		ed = levenshtein_dist(text_gt, text_pred)
		cer = float(ed / max(1, len(text_gt)))
		elapsed = time.time() - t_item
		return (idx, p_name, s_name, cer, ed, elapsed)

	indexed_items = list(enumerate(valid_pairs, 1))

	if engine.lower() == 'tesseract' and num_workers > 1:
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			results = list(executor.map(process_item, indexed_items))
	else:
		results = [process_item(item) for item in indexed_items]

	results.sort(key=lambda x: x[0])
	for idx, p_name, s_name, cer, ed, elapsed in results:
		cer_list.append(cer)
		ed_list.append(ed)
		time_list.append(elapsed)
		records.append([idx, p_name, s_name, f"{cer:.4f}", ed, f"{elapsed:.2f}"])
		print(f"{idx:<5} | {p_name:<25} | {s_name:<12} | {cer:<8.4f} | {ed:<7} | {elapsed:<8.2f}")

	mean_cer = float(np.mean(cer_list))
	mean_ed = float(np.mean(ed_list))
	mean_time = float(np.mean(time_list))
	total_time = time.time() - t_start_all

	print('=' * 85)
	print('OVERALL OCR EVALUATION RESULTS:')
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
		print(f'Saved OCR report to CSV: {output_csv}')

	return {
		'count': total,
		'mean_cer': mean_cer,
		'mean_ed': mean_ed,
		'mean_time': mean_time,
		'total_time': total_time,
		'records': records
	}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Official OCR Evaluation (CER & ED) for Document Dewarping')
	parser.add_argument('--pred_dir', type=str, required=True, help='Directory of dewarped images')
	parser.add_argument('--gt_dir', type=str, required=True, help='Directory of ground truth scan images')
	parser.add_argument('--output_csv', type=str, default=None, help='Path to output CSV report')
	parser.add_argument('--cache_json', type=str, default=None, help='Path to cache GT OCR JSON file')
	parser.add_argument('--benchmark', type=str, default='docunet_paper_subset', choices=['docunet_paper_subset', 'docunet_paper_70', 'docunet_130', 'all'])
	parser.add_argument('--engine', type=str, default='tesseract', choices=['tesseract', 'rapidocr'])
	parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
	args = parser.parse_args()

	evaluate_ocr(
		pred_dir=args.pred_dir,
		gt_dir=args.gt_dir,
		output_csv=args.output_csv,
		cache_json=args.cache_json,
		benchmark_name=args.benchmark,
		engine=args.engine,
		num_workers=args.workers
	)
