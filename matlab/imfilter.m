function retval = imfilter(A, h, varargin)
if nargin < 2
	error('imfilter: requires at least two input arguments.');
end
if isempty(A)
	retval = A;
	return;
end
orig_class = class(A);
is_uint8 = strcmp(orig_class, 'uint8');
dims = size(A);
rows = dims(1);
cols = dims(2);
if length(dims) > 2
	channels = prod(dims(3:end));
else
	channels = 1;
end
pad_mode = 0;
res_size = 'same';
ftype = 'corr';
for i = 1:length(varargin)
	arg = varargin{i};
	if ischar(arg)
		if strcmpi(arg, 'replicate') || strcmpi(arg, 'symmetric') || strcmpi(arg, 'circular')
			pad_mode = lower(arg);
		elseif strcmpi(arg, 'same') || strcmpi(arg, 'full')
			res_size = lower(arg);
		elseif strcmpi(arg, 'corr') || strcmpi(arg, 'conv')
			ftype = lower(arg);
		end
	elseif isscalar(arg) && isnumeric(arg)
		pad_mode = double(arg);
	end
end
[frows, fcols] = size(h);
pad_r_pre = floor((frows - 1) / 2);
pad_r_post = (frows - 1) - pad_r_pre;
pad_c_pre = floor((fcols - 1) / 2);
pad_c_post = (fcols - 1) - pad_c_pre;
if length(dims) > 3
	A = reshape(A, rows, cols, channels);
end
if ischar(pad_mode)
	if strcmp(pad_mode, 'replicate')
		r_idx = [ones(1, pad_r_pre), 1:rows, rows * ones(1, pad_r_post)];
		c_idx = [ones(1, pad_c_pre), 1:cols, cols * ones(1, pad_c_post)];
		A_pad = A(r_idx, c_idx, :);
	elseif strcmp(pad_mode, 'symmetric')
		r_idx = [pad_r_pre:-1:1, 1:rows, rows:-1:rows-pad_r_post+1];
		c_idx = [pad_c_pre:-1:1, 1:cols, cols:-1:cols-pad_c_post+1];
		A_pad = A(r_idx, c_idx, :);
	elseif strcmp(pad_mode, 'circular')
		r_idx = [mod(-pad_r_pre:-1, rows) + 1, 1:rows, mod(0:pad_r_post-1, rows) + 1];
		c_idx = [mod(-pad_c_pre:-1, cols) + 1, 1:cols, mod(0:pad_c_post-1, cols) + 1];
		A_pad = A(r_idx, c_idx, :);
	else
		A_pad = A;
	end
else
	A_pad = zeros(rows + pad_r_pre + pad_r_post, cols + pad_c_pre + pad_c_post, channels, orig_class);
	if pad_mode ~= 0
		A_pad(:) = cast(pad_mode, orig_class);
	end
	A_pad(pad_r_pre+1:pad_r_pre+rows, pad_c_pre+1:pad_c_pre+cols, :) = A;
end
is_separable = false;
g_col = [];
g_row = [];
if frows == 1 && fcols > 1
	is_separable = true;
	g_col = 1.0;
	g_row = double(h(:))';
elseif frows > 1 && fcols == 1
	is_separable = true;
	g_col = double(h(:));
	g_row = 1.0;
elseif frows > 1 && fcols > 1
	[U, S, V] = svd(double(h));
	if S(1, 1) > 0 && (S(2, 2) / S(1, 1) < 1e-5)
		is_separable = true;
		g_col = U(:, 1) * sqrt(S(1, 1));
		g_row = V(:, 1)' * sqrt(S(1, 1));
		if sum(g_col) < 0
			g_col = -g_col;
			g_row = -g_row;
		end
	end
end
if strcmpi(ftype, 'corr')
	if is_separable
		g_col = g_col(end:-1:1);
		g_row = g_row(end:-1:1);
	else
		h = rot90(h, 2);
	end
end
retval = zeros(rows, cols, channels, orig_class);
if is_separable
	g_col = double(g_col(:));
	g_row = double(g_row(:))';
	for c = 1:channels
		slice = double(A_pad(:,:,c));
		f_slice = conv2(g_col, g_row, slice, 'valid');
		if is_uint8
			retval(:,:,c) = uint8(round(f_slice));
		else
			retval(:,:,c) = cast(f_slice, orig_class);
		end
	end
else
	h_dbl = double(h);
	for c = 1:channels
		slice = double(A_pad(:,:,c));
		f_slice = conv2(slice, h_dbl, 'valid');
		if is_uint8
			retval(:,:,c) = uint8(round(f_slice));
		else
			retval(:,:,c) = cast(f_slice, orig_class);
		end
	end
end
if length(dims) > 2
	retval = reshape(retval, dims);
end
end
