%% GET NOTE MATRICES
files = dir([pwd '/midi/*.mid']);
for i = 1:length(files)
    file = files(i).name;
    fName = strcat(pwd, '/midi/', file);
    filename = strcat(pwd, '/', file(1:length(file) - 4), '_nm');
    nm = readmidi_java(fName);
    save(filename, 'nm');
end