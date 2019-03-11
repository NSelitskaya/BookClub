function [imageDS, dataSetFolders] = createBCbaselineIDS2d(dataFolderTmpl, dataFolderSfx, readFcn)

% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create a vector of the baseline sets
% Empty vector
dataSetFolders = strings(0);
labels = strings(0);

% Let's populate the vector by the baseline folder templates, one by one
dataSetFolders = [dataSetFolders, 'S1_Sfx/S1NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(1), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 

dataSetFolders = [dataSetFolders, 'S2_Sfx/S2NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(2), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S3_Sfx/S3NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(3), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S4_Sfx/S4NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(4), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S5_Sfx/S5NM4_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(5), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S6_Sfx/S6NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(6), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S7_Sfx/S7NM4_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(7), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S8_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(8), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S9_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(9), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S10_Sfx/S10NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(10), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S11_Sfx/S11NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(11), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S12_Sfx/S12NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(12), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S13_Sfx/S13NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(13), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S14_Sfx/S14NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(14), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S15_Sfx/S15NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(15), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S16_Sfx/S16NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(16), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S17_Sfx/S17NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(17), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S18_Sfx/S18NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(18), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S19_Sfx/S19NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(19), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S20_Sfx/S20NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(20), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S21_Sfx/S21NM1_Sfx'];
[tmpStr, ~] = strsplit(dataSetFolders(21), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];

%% Replace Sfx template with the actual value
dataSetFolders = strrep(dataSetFolders, 'Sfx', dataFolderSfx);
labels = strrep(labels, 'Sfx', dataFolderSfx);

% Build a full path  
fullDataSetFolders = fullfile(dataFolder, dataSetFolders);


[~, m] = size(labels);


%% Create a Datastore of the baseline images with individual folder labels
imageDS = imageDatastore(fullDataSetFolders, 'IncludeSubfolders', false,...
                            'LabelSource', 'none');                        
imageDS.ReadFcn = readFcn;


%% Label images by top subject's folder name                        
[n, ~] = size(imageDS.Files);  
labelStr = strings(n,1);

for j=1:m
      
    matches = contains(string(imageDS.Files), labels(j));
    labelStr(matches) = labels(j); 
    
end

imageDS.Labels = categorical(labelStr);

end

