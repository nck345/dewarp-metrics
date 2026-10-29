function B = impyramid(A, direction)
% Image pyramid reduction or expansion
% Uses Burt & Adelson 5x5 separable binomial filter
% Compatible with MATLAB and GNU Octave.

if strcmpi(direction, 'reduce')
    w = [1 4 6 4 1] / 16;
    kernel = w' * w;
    A_double = double(A);
    channels = size(A, 3);
    if channels == 1
        filtered = imfilter(A_double, kernel, 'same', 'replicate');
        B = filtered(1:2:end, 1:2:end);
    else
        filtered = zeros(size(A_double));
        for c = 1:channels
            filtered(:,:,c) = imfilter(A_double(:,:,c), kernel, 'same', 'replicate');
        end
        B = filtered(1:2:end, 1:2:end, :);
    end
    if isa(A, 'uint8')
        B = uint8(round(B));
    end
elseif strcmpi(direction, 'expand')
    error('Expand direction not needed for dewarping evaluation.');
else
    error('Direction must be ''reduce'' or ''expand''.');
end

end

