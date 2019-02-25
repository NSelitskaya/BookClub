function [outputArg1,outputArg2] = createBCimageDataStore(inputArg1,inputArg2)

% Empty vector
dataSetFolder = strings(0);

% Let's populate the vector by the baseline folders one by one
% 
dataSetFolder = [dataSetFolder, 'S1_1072x712/S1NM2_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S2_1072x712/S2NM1_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S3_1072x712/S3NM2_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S4_1072x712/S4NM1_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S5_1072x712/S5NM2_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S6_1072x712/S6NM1_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S7_1072x712/S7NM2_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S8_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S9_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S10_1072x712/S10NM1_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S11_1072x712/S11NM1_1072x712'];

%
dataSetFolder = [dataSetFolder, 'S12_1072x712/S12NM1_1072x712'];
 
%
dataSetFolder = [dataSetFolder, 'S13_1072x712/S13NM1_1072x712'];


% Count number of the folders
[~, nClasses] = size(dataSetFolder);



%% Build a full path  
fullDataSetFolder = fullfile(dataFolder, dataSetFolder);

%% Create a Datastore of the baseline images with individual folder lables
allImages = imageDatastore(fullDataSetFolder, 'IncludeSubfolders', false,...
                            'LabelSource', 'foldernames');                       
allImages.ReadFcn = @readFunctionGray_n;
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

