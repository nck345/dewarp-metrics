# Dewarp-Metrics

Official evaluation metrics suite for document image dewarping, strictly matching the benchmark methodology of DocUNet, DewarpNet, and DocTr.

Supports both **GNU Octave** (for headless execution on Google Colab / Linux) and **MATLAB**, alongside official Python OCR evaluation tools.

---

## 1. Supported Metrics

| Metric | Full Name | Standard Implementation | Optimal |
| :--- | :--- | :--- | :--- |
| **MS-SSIM-M** | Multi-Scale SSIM (Masked) | 5-scale Gaussian pyramid + Background Masking | Higher (↑) |
| **LD-M** | Local Distortion (Masked) | MIT SIFT Flow dense correspondence + Masking | Lower (↓) |
| **CER** | Character Error Rate | Tesseract 5.0 OCR + Levenshtein Distance | Lower (↓) |
| **ED** | Edit Distance | Tesseract 5.0 OCR + Levenshtein Distance | Lower (↓) |

### Why Background Masking (-M) Is Mandatory

Dewarped documents contain unmapped black border regions (`RGB = 0`). Evaluating raw SSIM without background masking causes severe distortion:
1. Comparing an interior page against a black boundary lowers SSIM by 0.15 to 0.25.
2. Models with slightly different cropping or thicker black margins suffer non-linear rank inversion: a model with superior internal text dewarping can score worse than a blurry output with thinner margins.
3. The standard paper formula masks the ground truth scan using the dewarped border mask (`ref_msk(A1 == 0) = 0`) and restricts SIFT Flow distortion computation to non-masked pixels (`d(~mskk)`).

---

## 2. Repository Structure

```
dewarp-metrics/
├── matlab/
│   ├── evalUnwarp.m           # Core MS-SSIM-M and LD-M calculator
│   ├── run_eval_octave.m      # Headless Octave/MATLAB CLI driver
│   ├── ssim.m                 # Wang et al. 2004 reference SSIM
│   ├── impyramid.m            # Gaussian pyramid reduction filter
│   └── SIFTflow/              # MIT SIFT Flow source and MEX code
│       ├── mexDenseSIFT/      # Dense SIFT MEX source
│       └── mexDiscreteFlow/   # Discrete flow matching MEX source
├── python/
│   └── ocr_eval.py            # Official Tesseract 5.0 CER & ED evaluator
├── compile_mex.sh             # Linux / Colab compilation script (GNU Octave)
├── compile_mex.m              # MATLAB compilation script
├── run_eval.py                # Unified end-to-end evaluation runner
└── requirements.txt           # Python dependencies
```

---

## 3. Installation & Setup

### Linux / Google Colab (GNU Octave)

Install system dependencies:
```bash
apt-get update -qq
apt-get install -y octave octave-image liboctave-dev tesseract-ocr libtesseract-dev -qq
```

Install Python packages:
```bash
pip install -r requirements.txt
```

Compile SIFT Flow MEX binaries for GNU Octave:
```bash
bash compile_mex.sh
```

### MATLAB (Windows / Linux / macOS)

Open MATLAB, navigate to the repository root, and run:
```matlab
compile_mex
```

---

## 4. Usage

### Unified Python Pipeline (Image + OCR)

Run the full evaluation pipeline and generate a combined CSV report:

```bash
python run_eval.py \
  --pred_dir "/path/to/dewarped_images" \
  --gt_dir "/path/to/scan_images" \
  --model_name "DocTr" \
  --benchmark "docunet_paper_subset" \
  --output_csv "eval_reports/DocTr_report.csv" \
  --ocr_engine "tesseract"
```

Available arguments:
- `--benchmark`: Benchmark subset to evaluate:
  - `docunet_paper_subset`: 60 images corresponding to 30 paper scan IDs (Setting 1).
  - `docunet_paper_70`: 70 images list specified in original paper OCR scripts.
  - `docunet_130`: Full 130 images of DocUNet benchmark.
  - `all`: All image pairs found in directory.
- `--ocr_engine`: `tesseract` (default, paper standard) or `rapidocr` (GPU alternative).
- `--skip_image`: Skip Octave/MATLAB image metrics.
- `--skip_ocr`: Skip Python OCR metrics.

### Standalone Image Evaluation (GNU Octave / MATLAB)

Run headless via command line:
```bash
octave --no-gui --silent --eval "addpath('matlab'); run_eval_octave('/path/to/preds', '/path/to/gt', 'report.csv', 'docunet_paper_subset'); exit;"
```

Or call directly within MATLAB:
```matlab
addpath('matlab');
run_eval_octave('path/to/preds', 'path/to/gt', 'report.csv', 'docunet_paper_subset');
```

### Standalone OCR Evaluation (Python)

```bash
python python/ocr_eval.py \
  --pred_dir "/path/to/dewarped_images" \
  --gt_dir "/path/to/scan_images" \
  --benchmark "docunet_paper_subset" \
  --engine "tesseract" \
  --output_csv "ocr_report.csv"
```

---

## 5. Benchmark Settings Reference

Standard 30 scan IDs for `docunet_paper_subset` (Setting 1):
`1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 24, 27, 30, 31, 32, 36, 38, 40, 41, 44, 45, 46, 47, 48, 50, 51, 52, 53`

Each scan corresponds to 2 warped test instances (`_1` and `_2`), yielding exactly 60 test images.
Target scan area for resolution normalization: `598400` pixels (~800 x 748).

