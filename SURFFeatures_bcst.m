%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC1_Sfx';
%dataFolderSfx = '96x64';
dataFolderSfx = '1072x712';

%% Folder for output files
outFolder = '~/data/BCTutorialOut';


%% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS2(dataFolderTmpl,...
                            dataFolderSfx, @readFunctionGray_n);
                        
%% Split Database into Training & Test Sets in the ratio 80% to 20%
%  Uncomment to test classifier on subsets of no-makeup images
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.8, 'randomize'); 

%%
trainingSet = baseSet;


% Count number of the classes ('stable' - presrvation of the order - just
% in case if we need it later)
labels = unique(baseSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)
             

%% Detect features on the trainingSet and build basis (vocabulary) of the bag

bag = bagOfFeatures(trainingSet,...
                    'PointSelection', 'Detector',...
                    'Upright', false, 'VocabularySize', 5000,...
                    'StrongestFeatures', 0.8, 'UseParallel', true);

                                    
%% Train the BOF classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag,...
                    'UseParallel', true);

                 
%% Evaluate the classifier on the test set images and display the confusion matrix
%  Uncomment to test classifier on no-makeup images
%[confMatrixTest, knownLabelIdx, predictedLabelIdx, score] =...
%           evaluate(categoryClassifier, testSet);
%        
% Compute average accuracy
%meanAcc = mean(diag(confMatrixTest));
%    
% Pre-extract SURF features and slice training set by labels once,
% without doing that for each test set when finding matches below
%[trainSets, trainFeatures, trainMetrics] = preextractSURFFeatures(bag, trainingSet);
%
% Iterate throug makeup test sets and find images with best (most numerous)
% matches between each test set and training sub-sets sliced by labels
%for i=1:nMakeups           
%    showSURFFeatureDRMatchesConfusion(bag, trainingSet, trainSets, trainFeatures,...
%        trainMetrics, testSets{i}, categoryClassifier,...
%        predictedLabelIdx{i}, outFolder, mkTable(i,:));
%end 
                            
 
%% Makeup datasets  
% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect2DR(dataFolderTmpl,...
                                dataFolderSfx, @readFunctionGray_n);


%% Allocate Confusion an Predicted Indexes matrixes of the needed size
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);
predictedLabelIdx = cell(nMakeups, 1);
meanMkAcc = zeros(nMakeups,1);
testDataSetLabels = strings(nMakeups, 1);

%% Run classifiers in parallel for each makeup test set
parfor i=1:nMakeups    

    % Extract true label of the test subset (from the first full file name)
    [tmpStr, ~] = strsplit(testSets{i}.Files{1,1}, '/');
    [~, nMatches] = size(tmpStr);
    mkLabel = tmpStr{1, nMatches-1};
    fprintf("Processing %s makeup set", mkLabel);
    
    
    %% Evaluate the classifier on the test set images   
    [predictedLabelIdx{i}, score] = predict(categoryClassifier, testSets{i});
       
    
    %% Compute average accuracy
    meanMkAcc(i) = mean(string(categoryClassifier.Labels(predictedLabelIdx{i})') == string(testSets{i}.Labels));
    

    %% Compute a row of the Confusion matrix.
    %  For current test set against all class labels
    [nFiles, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);
     
    maxAccCat = '';
    maxAcc = 0;
      
    for j = 1:nClasses
    
        meanMkConf(j) = mean( string(categoryClassifier.Labels(predictedLabelIdx{i})') == string(labels(j)) );
        
        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = string(labels(j)); %tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end
    
    % Update confusion matrix row at once (parfor requirement)
    mkTable(i,:) = num2cell([testDataSetFolders(i) meanMkAcc(i) maxAccCat maxAcc meanMkConf]);
    testDataSetLabels(i) = testSets{i}.Labels(1);
    
end

%% Confusion Matrix
varNames = cellstr(['TestFolder' 'Accuracy' 'BestGuess' 'GuessScore' string(labels)']);
cell2table(mkTable, 'VariableNames', varNames)
fprintf("Average imagewise accuracy %f\n", mean(meanMkAcc));
fprintf("Average sessionwise accuracy %f\n", mean(testDataSetLabels' == [mkTable{:,3}]));

%% Pre-extract SURF features and slice training set by labels once,
% without doing that for each test set when finding matches below
[trainSets, trainFeatures, trainMetrics] = preextractSURFFeatures(bag, trainingSet);

%% Iterate throug makeup test sets and find images with best (most numerous)
% matches between each test set and training sub-sets sliced by labels
for i=1:nMakeups           

    showSURFFeatureMatchesConfusion(bag, trainingSet, trainSets, trainFeatures,...
        trainMetrics, testSets{i}, categoryClassifier,...
        predictedLabelIdx{i}, outFolder, mkTable(i,:));

 end    
