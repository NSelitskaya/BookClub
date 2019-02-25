% Convert BC1_1072x712 data set images to dimensions set newDim.
% Create new folder structure BC1_XxY in the same root directory.
% 
% Natalya Selitskaya, Stanislaw Sielicki
% 01/25/2019

%% Clear everything 
clear all; close all; clc;

% Your folder (in Unix notation)
dataFolder = '~/data/BC1_1072x712';   

% Your new dimensions [Y X]
newDim = [64 96];
newDimStr = strcat( int2str(newDim(2)), 'x', int2str(newDim(1)));


%% Create a DataStore object that handless images in 
% the specified folder hierarchy as a whole
allImages = imageDatastore(dataFolder, 'IncludeSubfolders', true,...
                            'LabelSource', 'foldernames');                  

[n,~] = size(allImages.Files);

% Iterate through all files in the DataStore
for i = 1:n
    
    % Read the image and resize it
    [img, info] = readimage(allImages, i);
    imgR = imresize(img, newDim);
    
    
    % Create a new full file name for the resized image
    % with new dimension suffixes
    dataFolders = split(info.Filename, '/');    
    newDataFolders = strrep(dataFolders, '1072x712', newDimStr);
    
    [m,~] = size(newDataFolders);

    % Iterate through all subfolders in the newly created full 
    % file name and create missing subfolders on the disk (if needed)
    folder2check = string(newDataFolders(1));
    for j = 2:m-1
        folder2check = strcat( folder2check, '/', string(newDataFolders(j)));

        if ~exist(folder2check, 'dir')
            folder2check
            mkdir(folder2check);
        end
    
    end
    
    % Write the new reformatted image into folder hierarchy 
    % we just created
    newFN = strrep(info.Filename, '1072x712', newDimStr);
    imwrite(imgR, newFN);
end                