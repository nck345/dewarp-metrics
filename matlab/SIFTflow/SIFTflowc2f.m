% function to do coarse to fine SIFT flow matching
function [vx,vy,energylist,pyrd2]=SIFTflowc2f(im1,im2,SIFTflowpara,isdisplay,Segmentation,pyrd2_in)
if isfield(SIFTflowpara,'alpha')
	alpha=SIFTflowpara.alpha;
else
	alpha=0.01;
end
if isfield(SIFTflowpara,'d')
	d=SIFTflowpara.d;
else
	d=alpha*20;
end
if isfield(SIFTflowpara,'gamma')
	gamma=SIFTflowpara.gamma;
else
	gamma=0.001;
end
if isfield(SIFTflowpara,'nlevels')
	nlevels=SIFTflowpara.nlevels;
else
	nlevels=4;
end
if isfield(SIFTflowpara,'wsize')
	wsize=SIFTflowpara.wsize;
else
	wsize=3;
end
if isfield(SIFTflowpara,'topwsize')
	topwsize=SIFTflowpara.topwsize;
else
	topwsize=10;
end
if isfield(SIFTflowpara,'nIterations')
	nIterations=SIFTflowpara.nIterations;
else
	nIterations=40;
end
if isfield(SIFTflowpara,'nTopIterations')
	nTopIterations=SIFTflowpara.nTopIterations;
else
	nTopIterations=100;
end
if exist('isdisplay','var')~=1
	isdisplay=false;
end
if exist('Segmentation','var')==1 && ~isempty(Segmentation)
	IsSegmentation=true;
else
	IsSegmentation=false;
end
pyrd(1).im1=im1;
pyrd(1).im2=im2;
if IsSegmentation
	pyrd(1).seg=Segmentation;
end
has_pyrd2 = (nargin >= 6 && ~isempty(pyrd2_in));
if has_pyrd2
	for lvl = 1:nlevels
		pyrd(lvl).im2 = pyrd2_in(lvl).im2;
	end
end
h_gauss = fspecial('gaussian', 5, 0.67);
for i=2:nlevels
	try
		filt1 = imfilter(pyrd(i-1).im1, h_gauss, 'same', 'replicate');
	catch
		filt1 = filter_channels_replicate(pyrd(i-1).im1, h_gauss);
	end
	pyrd(i).im1 = imresize(filt1, 0.5, 'bicubic');
	if ~has_pyrd2
		try
			filt2 = imfilter(pyrd(i-1).im2, h_gauss, 'same', 'replicate');
		catch
			filt2 = filter_channels_replicate(pyrd(i-1).im2, h_gauss);
		end
		pyrd(i).im2 = imresize(filt2, 0.5, 'bicubic');
	end
	if IsSegmentation
		pyrd(i).seg=imresize(pyrd(i-1).seg,0.5,'nearest');
	end
end
for i=1:nlevels
	[height,width,nchannels]=size(pyrd(i).im1);
	[height2,width2,nchannels]=size(pyrd(i).im2);
	[xx,yy]=meshgrid(1:width,1:height);    
	pyrd(i).xx=round((xx-1)*(width2-1)/(width-1)+1-xx);
	pyrd(i).yy=round((yy-1)*(height2-1)/(height-1)+1-yy);
end
nIterationArray=round(linspace(nIterations,nIterations,nlevels));
for i=nlevels:-1:1
	if isdisplay
		fprintf('Level: %d...',i);
	end
	[height,width,nchannels]=size(pyrd(i).im1);
	[height2,width2,nchannels]=size(pyrd(i).im2);
	[xx,yy]=meshgrid(1:width,1:height);
	if i==nlevels
		vx=pyrd(i).xx;
		vy=pyrd(i).yy;
		winSizeX=ones(height,width)*topwsize;
		winSizeY=ones(height,width)*topwsize;
	else
		vx=round(pyrd(i).xx+imresize(vx-pyrd(i+1).xx,[height,width],'bicubic')*2);
		vy=round(pyrd(i).yy+imresize(vy-pyrd(i+1).yy,[height,width],'bicubic')*2);
		winSizeX=ones(height,width)*(wsize+i-1);
		winSizeY=ones(height,width)*(wsize+i-1);
	end
	if nchannels<=3
		Im1=im2feature(pyrd(i).im1);
		Im2=im2feature(pyrd(i).im2);
	else
		Im1=pyrd(i).im1;
		Im2=pyrd(i).im2;
	end
	if IsSegmentation
		imdiff=zeros(height,width,2);
		imdiff(:,1:end-1,1)=double(pyrd(i).seg(:,1:end-1)==pyrd(i).seg(:,2:end));
		imdiff(1:end-1,:,2)=double(pyrd(i).seg(1:end-1,:)==pyrd(i).seg(2:end,:));
		Im_s=imdiff*alpha+(1-imdiff)*alpha*0.01;
		Im_d=imdiff*alpha*100+(1-imdiff)*alpha*0.01*20;
	end
	if i==nlevels
		if IsSegmentation
			[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,2,topwsize],vx,vy,winSizeX,winSizeY,Im_s,Im_d);
		else
			[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,2,topwsize],vx,vy,winSizeX,winSizeY);
		end
	else
		if IsSegmentation
			[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),nlevels-i,wsize],vx,vy,winSizeX,winSizeY,Im_s,Im_d);
		else
			[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),nlevels-i,wsize],vx,vy,winSizeX,winSizeY);
		end
	end
	energylist(i).data=foo;
	vx=flow(:,:,1);
	vy=flow(:,:,2);
end
pyrd2 = struct();
for lvl = 1:nlevels
	pyrd2(lvl).im2 = pyrd(lvl).im2;
end
end

function winSizeIm=decideWinSize(offset,wsize)
f1=fspecial('gaussian',9,1);
f2=fspecial('gaussian',9,.5);
f=f2-f1;
foo=imfilter(abs(imfilter(offset,f,'same','replicate')),fspecial('gaussian',9,1.5),'same','replicate');
Min=min(foo(:));
Max=max(foo(:));
winSizeIm=wsize*(foo-Min)/(Max-Min)+wsize;
end

function out = filter_channels_replicate(im, h)
orig_class = class(im);
is_uint8 = strcmp(orig_class, 'uint8');
dims = size(im);
rows = dims(1);
cols = dims(2);
channels = 1;
if length(dims) > 2
	channels = prod(dims(3:end));
end
[frows, fcols] = size(h);
pad_r_pre = floor((frows - 1) / 2);
pad_r_post = (frows - 1) - pad_r_pre;
pad_c_pre = floor((fcols - 1) / 2);
pad_c_post = (fcols - 1) - pad_c_pre;
if length(dims) > 3
	im = reshape(im, rows, cols, channels);
end
r_idx = [ones(1, pad_r_pre), 1:rows, rows * ones(1, pad_r_post)];
c_idx = [ones(1, pad_c_pre), 1:cols, cols * ones(1, pad_c_post)];
im_pad = im(r_idx, c_idx, :);
is_sep = false;
g_col = [];
g_row = [];
if frows == 1 && fcols > 1
	is_sep = true;
	g_col = 1.0;
	g_row = double(h(:))';
elseif frows > 1 && fcols == 1
	is_sep = true;
	g_col = double(h(:));
	g_row = 1.0;
elseif frows > 1 && fcols > 1
	[U, S, V] = svd(double(h));
	if S(1, 1) > 0 && (S(2, 2) / S(1, 1) < 1e-5)
		is_sep = true;
		g_col = U(:, 1) * sqrt(S(1, 1));
		g_row = V(:, 1)' * sqrt(S(1, 1));
		if sum(g_col) < 0
			g_col = -g_col;
			g_row = -g_row;
		end
	end
end
out = zeros(rows, cols, channels, orig_class);
if is_sep
	g_col = double(g_col(end:-1:1));
	g_row = double(g_row(end:-1:1));
	g_col = g_col(:);
	g_row = g_row(:)';
	for c = 1:channels
		slice = double(im_pad(:,:,c));
		f_slice = conv2(g_col, g_row, slice, 'valid');
		if is_uint8
			out(:,:,c) = uint8(round(f_slice));
		else
			out(:,:,c) = cast(f_slice, orig_class);
		end
	end
else
	h_rot = double(rot90(h, 2));
	for c = 1:channels
		slice = double(im_pad(:,:,c));
		f_slice = conv2(slice, h_rot, 'valid');
		if is_uint8
			out(:,:,c) = uint8(round(f_slice));
		else
			out(:,:,c) = cast(f_slice, orig_class);
		end
	end
end
if length(dims) > 2
	out = reshape(out, dims);
end
end
