% Workspace startup for Benchmark_Reservoir
% This script configures MATLAB paths and optional Python interop.

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot)
    projectRoot = pwd;
end

% Add MATLAB source tree used in this repository.
matlabRoot = fullfile(projectRoot, ...
    'Model-free prediction of chaotic dynamics with parameter-aware reservoir', ...
    'Original');

if exist(matlabRoot, 'dir') == 7
    addpath(genpath(matlabRoot));
end

% Optional: bind MATLAB Python interface to workspace .venv.
venvPython = fullfile(projectRoot, '.venv', 'bin', 'python');
if exist(venvPython, 'file') == 2
    try
        pe = pyenv;
        if strcmp(pe.Status, 'Loaded')
            if ~strcmp(pe.Executable, venvPython)
                warning(['Python is already loaded: ', pe.Executable, ...
                    '. Restart MATLAB to switch to ', venvPython, '.']);
            end
        else
            pyenv('Version', venvPython);
        end
    catch ME
        warning(['Failed to configure pyenv: ', ME.message]);
    end
end

disp(['[startup] MATLAB project initialized at ', projectRoot]);
