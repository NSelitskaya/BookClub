function [imageDS, dataSetFolders] = createBCbaselineIDS(dataFolderTmpl, dataFolderSfx, readFcn)

%% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create a vector of the baseline sets
% Empty vector
dataSetFolders = strings(0);

% Let's populate the vector by the baseline folder templates, one by one
dataSetFolders = [dataSetFolders, 'S1_Sfx/S1NM2_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S2_Sfx/S2NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S3_Sfx/S3NM2_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S4_Sfx/S4NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S5_Sfx/S5NM2_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S6_Sfx/S6NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S7_Sfx/S7NM2_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S8_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S9_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S10_Sfx/S10NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S11_Sfx/S11NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S12_Sfx/S12NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S13_Sfx/S13NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S14_Sfx/S14NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S15_Sfx/S15NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S16_Sfx/S16NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S17_Sfx/S17NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S18_Sfx/S18NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S19_Sfx/S19NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S20_Sfx/S20NM1_Sfx'];
%
dataSetFolders = [dataSetFolders, 'S21_Sfx/S21NM1_Sfx'];

%% Replace Sfx template with the actual value
dataSetFolders = strrep(dataSetFolders, 'Sfx', dataFolderSfx);

% Build a full path  
fullDataSetFolders = fullfile(dataFolder, dataSetFolders);


%% Create a Datastore of the baseline images with individual folder labels
imageDS = imageDatastore(fullDataSetFolders, 'IncludeSubfolders', false,...
                            'LabelSource', 'foldernames');
                        
imageDS.ReadFcn = readFcn;

end

