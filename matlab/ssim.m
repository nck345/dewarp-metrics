function [mssim, ssim_map] = ssim(img1, img2, K, window, L)
% Standard Structural SIMilarity (SSIM) Index
% Based on Zhou Wang et al. 2004, IEEE TIP.
% Compatible with MATLAB and GNU Octave.

if nargin < 2
    error('ssim requires at least two input images.');
end

if any(size(img1) ~= size(img2))
    error('Input images must have the same size.');
end

if nargin < 3 || isempty(K)
    K = [0.01 0.03];
end

if nargin < 4 || isempty(window)
    window = fspecial('gaussian', 11, 1.5);
end

if nargin < 5 || isempty(L)
    L = 255;
end

img1 = double(img1);
img2 = double(img2);

if max(img1(:)) <= 1.0 && max(img2(:)) <= 1.0 && L == 255
    L = 1.0;
end

C1 = (K(1) * L)^2;
C2 = (K(2) * L)^2;

window = window / sum(window(:));

mu1 = imfilter(img1, window, 'replicate', 'same');
mu2 = imfilter(img2, window, 'replicate', 'same');

mu1_sq = mu1 .* mu1;
mu2_sq = mu2 .* mu2;
mu1_mu2 = mu1 .* mu2;

sigma1_sq = imfilter(img1 .* img1, window, 'replicate', 'same') - mu1_sq;
sigma2_sq = imfilter(img2 .* img2, window, 'replicate', 'same') - mu2_sq;
sigma12 = imfilter(img1 .* img2, window, 'replicate', 'same') - mu1_mu2;

if C1 > 0 && C2 > 0
    ssim_map = ((2 * mu1_mu2 + C1) .* (2 * sigma12 + C2)) ./ ...
               ((mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2));
else
    num1 = 2 * mu1_mu2 + C1;
    num2 = 2 * sigma12 + C2;
    den1 = mu1_sq + mu2_sq + C1;
    den2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(img1));
    index = (den1 .* den2 > 0);
    ssim_map(index) = (num1(index) .* num2(index)) ./ (den1(index) .* den2(index));
    index = (den1 ~= 0) & (den2 == 0);
    ssim_map(index) = num1(index) ./ den1(index);
end

f = floor(size(window) / 2);
valid_map = ssim_map(f(1)+1:end-f(1), f(2)+1:end-f(2));
mssim = mean(valid_map(:));

end

