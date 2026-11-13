function c=read_training_config()
root=fileparts(fileparts(mfilename('fullpath')));
lines=splitlines(string(fileread(fullfile(root,'configs','rl_training.ini'))));
section='';c=struct();
for line=lines'
    line=strtrim(line);if startsWith(line,{'#',';'})||strlength(line)==0,continue;end
    if startsWith(line,'['),section=char(extractBetween(line,'[',']'));continue;end
    parts=split(line,'=');if numel(parts)==2,c.(section).(strtrim(parts(1)))=str2double(parts(2));end
end
end
