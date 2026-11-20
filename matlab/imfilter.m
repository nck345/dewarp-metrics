function retval = imfilter(A, h, varargin)
% IMFILTER N-D array filtering compatible with MATLAB and GNU Octave.
% Supports arbitrary number of channels (1, 3, 128, etc.) and separable filtering.
% Fixes GNU Octave's native limitation where imfilter only accepts 1 or 3 channels.

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

% Parse options
pad_mode = 0; % default pad with 0
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
pad_r = floor(frows / 2);
pad_c = floor(fcols / 2);

% Check if filter is separable (rank 1)
is_separable = false;
g_col = [];
g_row = [];
if frows > 1 && fcols > 1
    [U, S, V] = svd(double(h));
    if S(2, 2) / S(1, 1) < 1e-5
        is_separable = true;
        g_col = U(:, 1) * sqrt(S(1, 1));
        g_row = V(:, 1)' * sqrt(S(1, 1));
        if sum(g_col) < 0
            g_col = -g_col;
            g_row = -g_row;
        end
    end
end

% If correlation requested, conv2 needs rotated kernel
% conv2(A, rot90(h, 2)) == corr(A, h)
if strcmpi(ftype, 'corr')
    if is_separable
        g_col = g_col(end:-1:1);
        g_row = g_row(end:-1:1);
    else
        h = rot90(h, 2);
    end
end

% Perform padding
if ischar(pad_mode)
    if strcmp(pad_mode, 'replicate')
        r_idx = [ones(1, pad_r), 1:rows, rows * ones(1, pad_r)];
        c_idx = [ones(1, pad_c), 1:cols, cols * ones(1, pad_c)];
        A_pad = A(r_idx, c_idx, :);
    elseif strcmp(pad_mode, 'symmetric')
        r_idx = [pad_r:-1:1, 1:rows, rows:-1:rows-pad_r+1];
        c_idx = [pad_c:-1:1, 1:cols, cols:-1:cols-pad_c+1];
        A_pad = A(r_idx, c_idx, :);
    elseif strcmp(pad_mode, 'circular')
        r_idx = [mod(-pad_r:-1, rows) + 1, 1:rows, mod(0:pad_r-1, rows) + 1];
        c_idx = [mod(-pad_c:-1, cols) + 1, 1:cols, mod(0:pad_c-1, cols) + 1];
        A_pad = A(r_idx, c_idx, :);
    else
        A_pad = A;
    end
else
    A_pad = zeros(rows + 2 * pad_r, cols + 2 * pad_c, channels, orig_class);
    if pad_mode ~= 0
        A_pad(:) = cast(pad_mode, orig_class);
    end
    A_pad(pad_r+1:pad_r+rows, pad_c+1:pad_c+cols, :) = A;
end

% Reshape for multi-channel processing if needed
if length(dims) > 3
    A_pad = reshape(A_pad, size(A_pad, 1), size(A_pad, 2), channels);
end

% Perform filtering
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
