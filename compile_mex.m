% Compile SIFT Flow MEX files in MATLAB
% Compatible with Windows, Linux, and macOS.

cur_dir = fileparts(mfilename('fullpath'));
mex_dense_dir = fullfile(cur_dir, 'matlab', 'SIFTflow', 'mexDenseSIFT');
mex_flow_dir = fullfile(cur_dir, 'matlab', 'SIFTflow', 'mexDiscreteFlow');

fprintf('=== Compiling SIFT Flow MEX for MATLAB ===\n');

fprintf('1/2 Compiling mexDenseSIFT in: %s\n', mex_dense_dir);
cd(mex_dense_dir);
mex mexDenseSIFT.cpp Matrix.cpp Vector.cpp

fprintf('2/2 Compiling mexDiscreteFlow in: %s\n', mex_flow_dir);
cd(mex_flow_dir);
mex mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp

cd(cur_dir);
fprintf('✅ MATLAB MEX compilation completed successfully!\n');
