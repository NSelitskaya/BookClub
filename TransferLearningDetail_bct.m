%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC1_Sfx';
%dataFolderSfx = '96x64';
dataFolderSfx = '1072x712';


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS3(dataFolderTmpl, dataFolderSfx, @readFunctionTrain_n);
trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - to use
% later for building confusion matrix)
labels = unique(baseSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)

                        
%% Split Database into Training & Test Sets in the ratio 80% to 20%
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.8, 'randomize'); 

        
%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
alex = alexnet; 

%% Review Network Architecture 
layers = alex.Layers;

%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
layers(23) = fullyConnectedLayer(nClasses); % change this based on # of classes
layers(25) = classificationLayer;

%% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 
opts = trainingOptions('sgdm',...
                       'ExecutionEnvironment','parallel',...
                       'InitialLearnRate', 0.001,...
                       'MiniBatchSize', 64);
                        
                      %'ExecutionEnvironment','parallel',...                          
                      %'MaxEpochs', 20,... 
                        
                      %'Plots', 'training-progress',...

%% Train the Network 
% This process usually takes about 5-20 minutes on a desktop GPU. 
myNet = trainNetwork(baseSet, layers, opts);
    
    
%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect3(dataFolderTmpl, dataFolderSfx, @readFunctionTrain_n);


%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%%
i = 1;
for i=1:nMakeups   
   
    
    %% Test Network Performance    
    predictedLabels = classify(myNet, testSets{i}); 
    
    
    %% Compute average accuracy
    meanMkAcc = mean(predictedLabels == testSets{i}.Labels);
    mkTable{i,1} = testDataSetFolders(i);
    mkTable{i,2} = meanMkAcc;
    
    %%
    [tn, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);

    maxAccCat = '';
    maxAcc = 0;
    
    %%    
    %labels = string(unique(allImages.Labels, 'stable'))';
    j = 1;   
    for j = 1:nClasses

        tmpStr = strings(tn,1);
        tmpStr(:) = string(labels(j));
    
        meanMkConf(j) = mean(string(predictedLabels) == tmpStr);
        mkTable{i, 4+j} = meanMkConf(j);
        
        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end
    mkTable{i,3} = maxAccCat;
    mkTable{i,4} = maxAcc;
    
end

%% Results
varNames = cellstr(['TestFolder' 'Accuracy' 'BestGuess' 'GuessScore' string(labels)']);
cell2table(mkTable, 'VariableNames', varNames)


%% Display poi and matches
%k = 1;
%img1 = readimage(mkTestSet, k);
%% Detect Features for First Image
%img1Pts = detectSURFFeatures(img1);
%[img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts);

%figure; imshow(img1);
%hold on; plot(img1Pts.selectStrongest(50));


%% find one image from the best matching category
%[n,~] = size(allImages.Files);
%for m = 1:n
    
%    [img2, info] = readimage(allImages, m);
%    if info.Label == maxAccCat
        %info.Label
%        break
%    end
%end

%% display matching features
%img2Pts = detectSURFFeatures(img2);
%[img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts);

%index_pairs = matchFeatures(img1Features, img2Features);

%img1Matched_pts = img1ValidPts(index_pairs(:,1)).Location;
%img2Matched_pts = img2ValidPts(index_pairs(:,2)).Location;

%figure, 
%showMatchedFeatures(img1, img2, img1Matched_pts,... 
%                        img2Matched_pts, 'montage');

%end %Makeups

%dataSetFolder
%fprintf("Training accuracy"); meanAccTrain
%fprintf("Test accuracy");meanAcc

%plotconfusion(categorical(knownLabelIdx), categorical(predictedLabelIdx));