import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image

class HighPrecisionMSSSIM(torch.nn.Module):
	"""
	High-Precision GPU-Accelerated MS-SSIM-M Calculator matching MATLAB Wang 2003/2004 standard.
	Uses torch.float64 (double precision) on CUDA/CPU with exact 5-scale Gaussian pyramid downsampling
	and document border background masking ref_msk(pred == 0) = 0.
	"""
	def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
		super().__init__()
		self.device = torch.device(device)

		# Gaussian 11x11, sigma=1.5 (Wang et al. 2004)
		size = 11
		sigma = 1.5
		coords = torch.arange(size, dtype=torch.float64, device=self.device) - (size - 1) / 2.0
		g = torch.exp(-(coords**2) / (2.0 * sigma**2))
		k_gauss = torch.outer(g, g)
		k_gauss = k_gauss / k_gauss.sum()
		self.window = k_gauss.unsqueeze(0).unsqueeze(0)

		# Burt & Adelson 5x5 separable binomial filter [1, 4, 6, 4, 1] / 16
		w = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64, device=self.device) / 16.0
		k_pyramid = torch.outer(w, w)
		self.pyramid_kernel = k_pyramid.unsqueeze(0).unsqueeze(0)

		self.k1 = 0.01
		self.k2 = 0.03
		self.L = 255.0
		self.c1 = (self.k1 * self.L)**2
		self.c2 = (self.k2 * self.L)**2
		self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64, device=self.device)

	def ssim_single_scale(self, x, y):
		x_pad = F.pad(x, (5, 5, 5, 5), mode='replicate')
		y_pad = F.pad(y, (5, 5, 5, 5), mode='replicate')

		mu1 = F.conv2d(x_pad, self.window)
		mu2 = F.conv2d(y_pad, self.window)
		mu1_sq = mu1 * mu1
		mu2_sq = mu2 * mu2
		mu1_mu2 = mu1 * mu2

		sigma1_sq = F.conv2d(x_pad * x_pad, self.window) - mu1_sq
		sigma2_sq = F.conv2d(y_pad * y_pad, self.window) - mu2_sq
		sigma12 = F.conv2d(x_pad * y_pad, self.window) - mu1_mu2

		ssim_map = ((2.0 * mu1_mu2 + self.c1) * (2.0 * sigma12 + self.c2)) / \
		           ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

		# Boundary exclusion matching Zhou Wang: 5 pixels on each side
		valid_map = ssim_map[:, :, 5:-5, 5:-5]
		return valid_map.mean()

	def reduce(self, x):
		x_pad = F.pad(x, (2, 2, 2, 2), mode='replicate')
		return F.conv2d(x_pad, self.pyramid_kernel, stride=2)

	@torch.no_grad()
	def forward(self, pred_tensor, ref_tensor):
		# pred_tensor, ref_tensor: [1, 1, H, W] in [0, 255] float64
		x = pred_tensor
		y = ref_tensor

		# Background masking: where dewarped image is black (0), mask GT to 0
		mask = (x == 0)
		z = y.clone()
		z[mask] = 0.0

		ss = torch.zeros(5, dtype=torch.float64, device=self.device)
		for s in range(5):
			ss[s] = self.ssim_single_scale(x, z)
			if s < 4:
				x = self.reduce(x)
				z = self.reduce(z)

		ms = (self.weights * ss).sum()
		return float(ms.item())

def evaluate_ms_ssim_pair(pred_path, scan_path, model=None, tarea=598400, device=None):
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if model is None:
		model = HighPrecisionMSSSIM(device=device)

	# Load images as grayscale float64
	pred_pil = Image.open(pred_path).convert('L')
	scan_pil = Image.open(scan_path).convert('L')

	s_w, s_h = scan_pil.size
	# Target scale matching MATLAB: b = sqrt(tarea / (H * W))
	b = np.sqrt(tarea / (s_h * s_w))
	target_h = int(round(s_h * b))
	target_w = int(round(s_w * b))

	# Anti-aliased bicubic resize matching MATLAB imresize
	scan_t = torch.from_numpy(np.array(scan_pil, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
	pred_t = torch.from_numpy(np.array(pred_pil, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

	scan_resized = TF.resize(scan_t, [target_h, target_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
	pred_resized = TF.resize(pred_t, [target_h, target_w], interpolation=InterpolationMode.BICUBIC, antialias=True)

	scan_d = scan_resized.to(device=device, dtype=torch.float64)
	pred_d = pred_resized.to(device=device, dtype=torch.float64)

	return model(pred_d, scan_d)
