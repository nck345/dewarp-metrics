function [ms, ld, sift2, im2] = evalUnwarp(A, ref, ref_msk, sift2_in, im2_in, skip_ms)
% EVALUNWARP Compute MS-SSIM-M and LD-M with background masking and optional SIFT caching
%   A:        rectified / unwarped image (grayscale)
%   ref:      reference / scan image (grayscale)
%   ref_msk:  masked reference image (zeros where A has black background borders)
%   sift2_in: (Optional) Precomputed dense SIFT descriptor for scan image
%   im2_in:   (Optional) Precomputed downsampled scan image
%   skip_ms:  (Optional) If true, skips MS-SSIM-M calculation (returns ms = 0)
%
%   ms: Multi-Scale SSIM with masking (MS-SSIM-M)
%   ld: Local Distortion with masking (LD-M)
%   sift2, im2: dense SIFT and downsampled scan for memory caching

if nargin < 4 || isempty(sift2_in) || isempty(im2_in)
    has_cached_sift = false;
else
    has_cached_sift = true;
end

if nargin < 6 || isempty(skip_ms)
    skip_ms = false;
end

x = A;
y = ref;
z = ref_msk;

im1 = imresize(imfilter(x, fspecial('gaussian', 7, 1.0), 'same', 'replicate'), 0.5, 'bicubic');
im3 = imresize(imfilter(z, fspecial('gaussian', 7, 1.0), 'same', 'replicate'), 0.5, 'bicubic');

im1 = im2double(im1);
im3 = im2double(im3);

cellsize = 3;
gridspacing = 1;

sift1 = mexDenseSIFT(im1, cellsize, gridspacing);

if has_cached_sift
    sift2 = sift2_in;
    im2 = im2_in;
else
    im2 = imresize(imfilter(y, fspecial('gaussian', 7, 1.0), 'same', 'replicate'), 0.5, 'bicubic');
    im2 = im2double(im2);
    sift2 = mexDenseSIFT(im2, cellsize, gridspacing);
end

SIFTflowpara.alpha = 2 * 255;
SIFTflowpara.d = 40 * 255;
SIFTflowpara.gamma = 0.005 * 255;
SIFTflowpara.nlevels = 4;
SIFTflowpara.wsize = 2;
SIFTflowpara.topwsize = 10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations = 30;

[vx, vy, ~] = SIFTflowc2f(sift1, sift2, SIFTflowpara);

d = sqrt(vx.^2 + vy.^2);
mskk = (im3 == 0);
valid_idx = ~mskk;

if any(valid_idx(:))
    ld = mean(d(valid_idx));
else
    ld = mean(d(:));
end

if skip_ms
    ms = 0;
else
    wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
    ss = zeros(5, 1);
    for s = 1 : 5
        ss(s) = ssim(x, z);
        x = impyramid(x, 'reduce');
        z = impyramid(z, 'reduce');
    end
    ms = wt * ss;
end

end
