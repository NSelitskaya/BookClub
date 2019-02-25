function [mkImages, mkDataSetFolders] = createBCtestIDSvect(dataFolderTmpl, dataFolderSfx, readFcn)

%% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create vectors of the makeup folder templates and category labels
% Empty vectors
mkDataSetFolders = strings(0);
mkLabels = strings(0);

% Let's populate vectors one by one, making labels from the top directory
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1GL1_Sfx'];
[tmpStr, ~] = strsplit(mkDataSetFolders(1), '/');
mkLabelCur = tmpStr(1,1);
mkLabels = [mkLabels, mkLabelCur];

mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1HD1_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1HD2_Sfx'];
mkLabels = [mkLabels, mkLabelCur]; 
mkDataSetFolders = [mkDataSetFolders, 'S1_Sfx/S1NM1_Sfx'];
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


%% Replace Sfx template with the actual value of the image dimensions
mkDataSetFolders = strrep(mkDataSetFolders, 'Sfx', dataFolderSfx);
mkLabels = strrep(mkLabels, 'Sfx', dataFolderSfx);

% Build a full path  
mkDataSetFullFolders = fullfile(dataFolder, mkDataSetFolders);


%% Create a vector of the makeup images Datastores with top folder labels
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
    mkImage.Labels = tmpStr; 
                            
    countEachLabel(mkImage)    
    
    mkImages{i} = mkImage;
end                        
    

end

