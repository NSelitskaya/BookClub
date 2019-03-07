%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC1_Sfx';
dataFolderSfx = '96x64';
%dataFolderSfx = '1072x712';

%% Folder for output files
outFolder = '~/data/BCTutorialOut';


%% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS2(dataFolderTmpl,...
                            dataFolderSfx, @readFunctionGray_n);
trainingSet = baseSet;


% Count number of the classes ('stable' - presrvation of the order - just
% in case if we need it later)
labels = unique(baseSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)

                            
%% Split Database into Training & Test Sets in the ratio 80% to 20%
%  Uncomment to test classifier on no-makeup images
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.8, 'randomize'); 


%% Detect features on the trainingSet and build basis (vocabulary) of the bag
bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector',...
                    'Upright', false, 'VocabularySize', 500,...
                    'StrongestFeatures', 0.8, 'UseParallel', true);

                                    
%% Train the BOF classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag,...
                     'UseParallel', true);

                 
%% Evaluate the classifier on the test set images and display the confusion matrix
%  Uncomment to test classifier on no-makeup images
%[confMatrixTest, knownLabelIdx, predictedLabelIdx, score] =...
%           evaluate(categoryClassifier, testSet);
        
% Compute average accuracy
%meanAcc = mean(diag(confMatrixTest));
    
%[trainFeatures, trainSets] = extractSURFFeatures(trainingSet);           
%showFeatureMatchesConfusion(trainingSet, trainSets, trainFeatures,...
%        testSet, categoryClassifier, predictedLabelIdx);
               
             
 
%% Makeup datasets  

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect2(dataFolderTmpl,...
                                dataFolderSfx, @readFunctionGray_n);


%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);
predictedLabelIdx = cell(nMakeups, 1);

%%
i = 1;
%par
for i=1:nMakeups    
   
    %% Evaluate the classifier on the test set images   
    [predictedLabelIdx{i}, score] = predict(categoryClassifier, testSets{i});
       
    
    %% Compute average accuracy
    meanMkAcc = mean(string(categoryClassifier.Labels(predictedLabelIdx{i})') == string(testSets{i}.Labels));
    %mkTable{i,1} = testDataSetFolders(i);
    %mkTable{i,2} = meanMkAcc;
    

    %% Compute a row of the Confusion matrix.
    %  For current test set against all class labels
    [nFiles, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);
     
    maxAccCat = '';
    maxAcc = 0;
    
    j = 1;   
    parfor j = 1:nClasses

        tmpStr = strings(nFiles,1);
        tmpStr(:) = string(labels(j));
    
        meanMkConf(j) = mean(string(categoryClassifier.Labels(predictedLabelIdx{i})') == tmpStr);
        %mkTable{i, 4+j} = meanMkConf(j);
        
        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end
    %mkTable{i,3} = maxAccCat;
    %mkTable{i,4} = maxAcc;
    
    mkTable(i,:) = num2cell([testDataSetFolders(i) meanMkAcc maxAccCat maxAcc meanMkConf]);
    
end

%% Confusion Matrix
varNames = cellstr(['TestFolder' 'Accuracy' 'BestGuess' 'GuessScore' string(labels)']);
cell2table(mkTable, 'VariableNames', varNames)


%%
%
[trainFeatures, trainSets] = extractSURFFeatures(trainingSet);

%%
i = 1;
for i=1:nMakeups           
    %
    showFeatureMatchesConfusion(trainingSet, trainSets, trainFeatures,...
        testSets{i}, categoryClassifier, predictedLabelIdx{i}, outFolder, mkTable(i,:));
       
end    