function [mkImages, mkDataSetFolders] = createBCtestIDSvect3(dataFolderTmpl, dataFolderSfx, readFcn)

%% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create vectors of the makeup folder templates and category labels

% Empty vectors
mkDataSetFolders = strings(0);
mkLabels = strings(0);

% Let's populate vectors one by one, making labels from the top directory

mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1GL1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1HD1_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1HD2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];  
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK1_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK3_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK4_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK5_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK6_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK7_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1MK8_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S2_Sfx/S2HD1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S2_Sfx/S2GL1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3HD1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3HD2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3MK1_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3MK3_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3MK4_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S3_Sfx/S3MK5_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S4_Sfx/S4HD1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S4_Sfx/S4GL1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S4_Sfx/S4MK1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S5_Sfx/S5MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S5_Sfx/S5MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S5_Sfx/S5MK3_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S5_Sfx/S5MK4_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S6_Sfx/S6MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S6_Sfx/S6MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S7_Sfx/S7MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S7_Sfx/S7MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S7_Sfx/S7MK3_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S7_Sfx/S7HD1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S7_Sfx/S7HD2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S10_Sfx/S10MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S10_Sfx/S10MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S10_Sfx/S10MK3_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S10_Sfx/S10MK4_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S11_Sfx/S11MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S11_Sfx/S11MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S12_Sfx/S12MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S12_Sfx/S12MK2_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S13_Sfx/S13MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S14_Sfx/S14HD1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S14_Sfx/S14GL1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S14_Sfx/S14MK1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S15_Sfx/S15MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S16_Sfx/S16MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S17_Sfx/S17MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S18_Sfx/S18MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S19_Sfx/S19GL1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S19_Sfx/S19MK1_Sfx'];
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S20_Sfx/S20MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S21_Sfx/S21MK1_Sfx'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];


%% Replace Sfx template with the actual value of the image dimensions
mkDataSetFolders = strrep(mkDataSetFolders, 'Sfx', dataFolderSfx);
mkLabels = strrep(mkLabels, 'Sfx', dataFolderSfx);

% Build a full path  
mkDataSetFullFolders = fullfile(dataFolder, mkDataSetFolders);


%% Create a vector of the makeup iamges Datastores with top folder lables

[~, nMakeups] = size(mkDataSetFolders);
mkImages = cell(nMakeups,1);


for i=1:nMakeups
    
    
    %% Create Datastore for each label
    mkImage = imageDatastore(mkDataSetFullFolders(i), 'IncludeSubfolders', false,...
                                'LabelSource', 'none');
    mkImage.ReadFcn = readFcn;
       
    % Label all images in the Datastore with the top folder label                       
    [n, ~] = size(mkImage.Files);  
    tmpStr = strings(n,1);
    tmpStr(:) = mkLabels(i);
    mkImage.Labels = categorical(tmpStr); 
                            
    countEachLabel(mkImage)    
    
    mkImages{i} = mkImage;
end                        
    

end

