function run_eval_octave(pred_dir, gt_dir, output_csv, benchmark_name, part_idx, total_parts, skip_ms)
% RUN_EVAL_OCTAVE Universal Headless Image Metrics Runner (MS-SSIM-M & LD-M)
% Compatible with GNU Octave (Google Colab / Linux / Windows) and MATLAB.

if nargin < 2
    error('Usage: run_eval_octave(pred_dir, gt_dir, [output_csv], [benchmark_name], [part_idx], [total_parts], [skip_ms])');
end

if nargin < 3 || isempty(output_csv)
    output_csv = '';
end

if nargin < 4 || isempty(benchmark_name)
    benchmark_name = 'all';
end

if nargin < 5 || isempty(part_idx)
    part_idx = 1;
end
if nargin < 6 || isempty(total_parts)
    total_parts = 1;
end
if nargin < 7 || isempty(skip_ms)
    skip_ms = false;
end

if ischar(part_idx)
    part_idx = str2double(part_idx);
end
if ischar(total_parts)
    total_parts = str2double(total_parts);
end
if ischar(skip_ms)
    skip_ms = strcmpi(skip_ms, 'true') || strcmp(skip_ms, '1');
end

% Setup paths
cur_dir = fileparts(mfilename('fullpath'));
addpath(cur_dir);
addpath(fullfile(cur_dir, 'SIFTflow'));
addpath(fullfile(cur_dir, 'SIFTflow', 'mexDenseSIFT'));
addpath(fullfile(cur_dir, 'SIFTflow', 'mexDiscreteFlow'));

% Octave image package load
if exist('OCTAVE_VERSION', 'builtin')
    try
        pkg load image;
    catch
    end
    more off;
end

% Check MEX binaries (3 = mex, 103 = oct)
if exist('mexDenseSIFT') ~= 3 && exist('mexDenseSIFT') ~= 103
    error('mexDenseSIFT MEX binary not found! Please run compile_mex script first.');
end
if exist('mexDiscreteFlow') ~= 3 && exist('mexDiscreteFlow') ~= 103
    error('mexDiscreteFlow MEX binary not found! Please run compile_mex script first.');
end

paper_ids = {'1', '2', '3', '4', '5', '6', '7', '9', '10', '21', ...
             '22', '23', '24', '27', '30', '31', '32', '36', '38', '40', ...
             '41', '44', '45', '46', '47', '48', '50', '51', '52', '53'};

raw_files = dir(pred_dir);
img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG'};
pred_list = {};

for k = 1:length(raw_files)
    if raw_files(k).isdir
        continue;
    end
    [~, ~, ext] = fileparts(raw_files(k).name);
    if ismember(ext, img_exts)
        pred_list{end+1} = raw_files(k).name;
    end
end

pred_list = sort(pred_list);

valid_pairs = {};
for k = 1:length(pred_list)
    p_name = pred_list{k};
    [~, name_no_ext, ~] = fileparts(p_name);
    parts = strsplit(name_no_ext, '_');
    doc_id = regexprep(parts{1}, '[^0-9]', '');
    if isempty(doc_id)
        doc_id = regexprep(name_no_ext, '[^0-9]', '');
    end

    if strcmpi(benchmark_name, 'docunet_paper_subset')
        if ~ismember(doc_id, paper_ids)
            continue;
        end
    end

    gt_file = '';
    scan_name = '';
    for e = 1:length(img_exts)
        candidate = fullfile(gt_dir, [doc_id, img_exts{e}]);
        if exist(candidate, 'file')
            gt_file = candidate;
            scan_name = [doc_id, img_exts{e}];
            break;
        end
    end
    if isempty(gt_file)
        for e = 1:length(img_exts)
            candidate = fullfile(gt_dir, [name_no_ext, img_exts{e}]);
            if exist(candidate, 'file')
                gt_file = candidate;
                scan_name = [name_no_ext, img_exts{e}];
                break;
            end
        end
    end

    if ~isempty(gt_file)
        valid_pairs{end+1} = struct('pred_name', p_name, ...
                                    'pred_path', fullfile(pred_dir, p_name), ...
                                    'scan_name', scan_name, ...
                                    'scan_path', gt_file);
    end
end

if total_parts > 1
    my_pairs = {};
    for k = 1:length(valid_pairs)
        if mod(k - 1, total_parts) + 1 == part_idx
            my_pairs{end+1} = valid_pairs{k};
        end
    end
    valid_pairs = my_pairs;
end

num_pairs = length(valid_pairs);
fprintf('========================================================================================\n');
if total_parts > 1
    fprintf('=== Official MATLAB/Octave Image Evaluation [Worker %d/%d] ===\n', part_idx, total_parts);
else
    fprintf('=== Official MATLAB/Octave Image Evaluation (SIFT Flow LD-M) ===\n');
end
fprintf('Benchmark       : %s\n', benchmark_name);
fprintf('Predictions dir : %s\n', pred_dir);
fprintf('Ground Truth dir: %s\n', gt_dir);
fprintf('Worker images   : %d\n', num_pairs);
fprintf('MS-SSIM-M mode  : %s\n', ternary(skip_ms, 'Skipped in Octave (Computed on GPU via PyTorch)', 'Computed in Octave'));
fprintf('========================================================================================\n');
if skip_ms
    fprintf('%-5s | %-25s | %-12s | %-8s | %-8s\n', 'STT', 'Dewarped Image', 'Scan GT', 'LD-M', 'Time (s)');
else
    fprintf('%-5s | %-25s | %-12s | %-9s | %-8s | %-8s\n', 'STT', 'Dewarped Image', 'Scan GT', 'MS-SSIM-M', 'LD-M', 'Time (s)');
end
fprintf('----------------------------------------------------------------------------------------\n');

if num_pairs == 0
    fprintf('No valid image pairs found.\n');
    return;
end

tarea = 598400;
ms_list = zeros(num_pairs, 1);
ld_list = zeros(num_pairs, 1);
time_list = zeros(num_pairs, 1);
results_cell = cell(num_pairs, 6);
scan_sift_cache = struct();

t_all_start = tic;

for i = 1:num_pairs
    t_img_start = tic;
    pair = valid_pairs{i};

    A1 = imread(pair.pred_path);
    ref = imread(pair.scan_path);

    if size(A1, 3) == 3
        A1 = rgb2gray(A1);
    end
    if size(ref, 3) == 3
        ref = rgb2gray(ref);
    end

    b = sqrt(tarea / size(ref, 1) / size(ref, 2));
    ref = imresize(ref, b);
    A1 = imresize(A1, [size(ref, 1), size(ref, 2)]);

    ref_msk = ref;
    m1 = (A1 == 0);
    ref_msk(m1) = 0;

    % Check in-memory scan SIFT descriptor cache
    scan_key = ['scan_', regexprep(pair.scan_name, '[^a-zA-Z0-9]', '_')];
    if isfield(scan_sift_cache, scan_key)
        sift2_c = scan_sift_cache.(scan_key).sift2;
        im2_c = scan_sift_cache.(scan_key).im2;
    else
        sift2_c = [];
        im2_c = [];
    end

    [ms_val, ld_val, sift2_out, im2_out] = evalUnwarp(A1, ref, ref_msk, sift2_c, im2_c, skip_ms);

    if isempty(sift2_c)
        scan_sift_cache.(scan_key).sift2 = sift2_out;
        scan_sift_cache.(scan_key).im2 = im2_out;
    end

    img_time = toc(t_img_start);

    ms_list(i) = ms_val;
    ld_list(i) = ld_val;
    time_list(i) = img_time;

    results_cell{i, 1} = i;
    results_cell{i, 2} = pair.pred_name;
    results_cell{i, 3} = pair.scan_name;
    results_cell{i, 4} = ms_val;
    results_cell{i, 5} = ld_val;
    results_cell{i, 6} = img_time;

    if skip_ms
        fprintf('%-5d | %-25s | %-12s | %-8.2f | %-8.2f\n', ...
                i, pair.pred_name, pair.scan_name, ld_val, img_time);
    else
        fprintf('%-5d | %-25s | %-12s | %-9.4f | %-8.2f | %-8.2f\n', ...
                i, pair.pred_name, pair.scan_name, ms_val, ld_val, img_time);
    end
    fflush(stdout);
end

t_all_elapsed = toc(t_all_start);
mean_ms = mean(ms_list);
mean_ld = mean(ld_list);
mean_time = mean(time_list);

fprintf('========================================================================================\n');
fprintf('OVERALL WORKER EVALUATION RESULTS (OCTAVE):\n');
fprintf('- Total evaluated images: %d\n', num_pairs);
if ~skip_ms
    fprintf('- MS-SSIM-M (↑)         : %.4f\n', mean_ms);
end
fprintf('- LD-M (↓)              : %.2f\n', mean_ld);
fprintf('- Mean time per image   : %.2f s\n', mean_time);
fprintf('- Total elapsed time    : %.1f s (%.2f min)\n', t_all_elapsed, t_all_elapsed / 60);
fprintf('========================================================================================\n');

if ~isempty(output_csv)
    [csv_dir, ~, ~] = fileparts(output_csv);
    if ~isempty(csv_dir) && exist(csv_dir, 'dir') ~= 7
        mkdir(csv_dir);
    end

    fid = fopen(output_csv, 'w');
    if fid ~= -1
        fprintf(fid, 'Index,Dewarped_Image,Scan_GT,MS_SSIM_M,LD_M,Time_seconds\n');
        for k = 1:num_pairs
            fprintf(fid, '%d,%s,%s,%.4f,%.4f,%.2f\n', ...
                    results_cell{k, 1}, results_cell{k, 2}, results_cell{k, 3}, ...
                    results_cell{k, 4}, results_cell{k, 5}, results_cell{k, 6});
        end
        fprintf(fid, 'SUMMARY,Total: %d,All,%.4f,%.4f,%.2f\n', ...
                num_pairs, mean_ms, mean_ld, mean_time);
        fclose(fid);
    end
end

end

function val = ternary(cond, true_val, false_val)
    if cond
        val = true_val;
    else
        val = false_val;
    end
end
