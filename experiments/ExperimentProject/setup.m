library_folders = dir(fullfile('..', '..', 'libraries'));
for i = 3:numel(library_folders)
    addpath(fullfile('..', '..', 'libraries', library_folders(i).name));
end
% addpath(fullfile('..', '..', '..', 'Documents/galahad'));
addpath(fullfile('.'));

python_executable = fullfile('..', '..', '.venv', 'bin', 'python');
% pyenv(Version = python_executable);
