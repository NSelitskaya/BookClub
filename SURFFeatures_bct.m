%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC1_Sfx';
%dataFolderSfx = '96x64';
dataFolderSfx = '1072x712';


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS3(dataFolderTmpl, dataFolderSfx, @readFunctionGray_n);
trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - just
% in case if we need it later)
labels = unique(baseSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)

                            
%% Split Database into Training & Test Sets in the ratio 80% to 20%
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.8, 'randomize'); 


%% Detect features on the trainingSet and build basis (vocabulary) of the bag
bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector',...
                    'Upright', false, 'VocabularySize', 500,...
                    'StrongestFeatures', 0.8, 'UseParallel', true);

                                    
%% Train the BOF classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag,...
                     'UseParallel', true);

                 
%% Evaluate the classifier on the test set images and display the confusion matrix
%[confMatrixTest, knownLabelIdx, predictedLabelIdx, score] =...
%           evaluate(categoryClassifier, testSet);
        
% Compute average accuracy
%meanAcc = mean(diag(confMatrixTest));
    
% Error details - which image is assigned to which class
%testLabels = categoryClassifier.Labels(predictedLabelIdx)';    
%M = [testSet.Files, testLabels]';
%fprintf("%s %s\n", M{:});



                 
             

%% Makeup datasets  

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect3(dataFolderTmpl, dataFolderSfx, @readFunctionGray_n);


%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%%
i = 1;
for i=1:nMakeups
    
   
    %% Evaluate the classifier on the test set images   
    [predictedLabelIdx, score] = predict(categoryClassifier, testSets{i});
       
    
    %% Compute average accuracy
    meanMkAcc = mean(string(categoryClassifier.Labels(predictedLabelIdx)') == string(testSets{i}.Labels));
    mkTable{i,1} = testDataSetFolders(i);
    mkTable{i,2} = meanMkAcc;
    
    %%
    [~, ~] = splitPredictions(testSets{i}, categoryClassifier, predictedLabelIdx, score);
    

    %%
    [tn, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);
     
    maxAccCat = '';
    maxAcc = 0;
    
    j = 1;   
    for j = 1:nClasses

        tmpStr = strings(tn,1);
        tmpStr(:) = string(labels(j));
    
        meanMkConf(j) = mean(string(categoryClassifier.Labels(predictedLabelIdx)') == tmpStr);
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



%%
mkDataSetFolder = strings(0);
mkLabel = strings(0);

[tmpStr, ~] = strsplit(dataSetFolder(1), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
%mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1GL1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1NM1_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK2_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK3_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK4_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK5_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S1_1072x712/S1MK6_1072x712'];
mkLabel = [mkLabel, mkLabelCur];


%[tmpStr, ~] = strsplit(dataSetFolder(2), '/');
%[~, c] = size(tmpStr);
%mkLabelCur = tmpStr(1,c);
%mkDataSetFolder = [mkDataSetFolder, 'S2_1072x712/S2GL1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];
%mkDataSetFolder = [mkDataSetFolder, 'S2_1072x712/S2HD1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];


[tmpStr, ~] = strsplit(dataSetFolder(3), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
%mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3HD1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];
%mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3HD2_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3NM1_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur]; 
mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3MK2_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3MK3_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S3_1072x712/S3MK4_1072x712'];
mkLabel = [mkLabel, mkLabelCur];


[tmpStr, ~] = strsplit(dataSetFolder(4), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
mkDataSetFolder = [mkDataSetFolder, 'S4_1072x712/S4MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
%mkDataSetFolder = [mkDataSetFolder, 'S4_1072x712/S4GL1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];
%mkDataSetFolder = [mkDataSetFolder, 'S4_1072x712/S4HD1_1072x712'];
%mkLabel = [mkLabel, mkLabelCur];


[tmpStr, ~] = strsplit(dataSetFolder(5), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
mkDataSetFolder = [mkDataSetFolder, 'S5_1072x712/S5NM1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S5_1072x712/S5NM3_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S5_1072x712/S5MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S5_1072x712/S5MK2_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S5_1072x712/S5MK3_1072x712'];
mkLabel = [mkLabel, mkLabelCur];


[tmpStr, ~] = strsplit(dataSetFolder(6), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
mkDataSetFolder = [mkDataSetFolder, 'S6_1072x712/S6MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S6_1072x712/S6MK2_1072x712'];
mkLabel = [mkLabel, mkLabelCur];


[tmpStr, ~] = strsplit(dataSetFolder(7), '/');
[~, c] = size(tmpStr);
mkLabelCur = tmpStr(1,c);
mkDataSetFolder = [mkDataSetFolder, 'S7_1072x712/S7NM1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S7_1072x712/S7MK1_1072x712'];
mkLabel = [mkLabel, mkLabelCur];
mkDataSetFolder = [mkDataSetFolder, 'S7_1072x712/S7MK2_1072x712'];
mkLabel = [mkLabel, mkLabelCur];


%[tmpStr, ~] = strsplit(dataSetFolder(10), '/');
%[~, c] = size(tmpStr);
%mkLabelCur = tmpStr(1,c);
%mkDataSetFolder = [mkDataSetFolder, 'S10_1072x712/S10MK1_1072x712'];
%mkLabel = [mkLabel, 'S10NM1_1072x712'];
%mkDataSetFolder = [mkDataSetFolder, 'S10_1072x712/S10MK2_1072x712'];
%mkLabel = [mkLabel, 'S10NM1_1072x712'];
%mkDataSetFolder = [mkDataSetFolder, 'S10_1072x712/S10MK3_1072x712'];
%mkLabel = [mkLabel, 'S10NM1_1072x712'];
%mkDataSetFolder = [mkDataSetFolder, 'S10_1072x712/S10MK4_1072x712'];
%mkLabel = [mkLabel, 'S10NM1_1072x712'];


%[tmpStr, ~] = strsplit(dataSetFolder(11), '/');
%[~, c] = size(tmpStr);
%mkLabelCur = tmpStr(1,c);
%mkDataSetFolder = [mkDataSetFolder, 'S11_1072x712/S11MK1_1072x712'];
%mkLabel = [mkLabel, 'S11NM1_1072x712'];
%mkDataSetFolder = [mkDataSetFolder, 'S11_1072x712/S11MK2_1072x712'];
%mkLabel = [mkLabel, 'S11NM1_1072x712'];


[~, nMakeups] = size(mkDataSetFolder);


%%
i = 2;
for i=1:nMakeups
    
    mkDataSetFullFolder = fullfile(dataFolder, mkDataSetFolder(i));
     
    %% Load Image Information from Makup Face Database
    mkImages = imageDatastore(mkDataSetFullFolder, 'IncludeSubfolders', false,...
                                'LabelSource', 'none');
    mkImages.ReadFcn = @readFunctionGray_n;
   
    % Label images                        
    [n, ~] = size(mkImages.Files);  
    tmpStr = strings(n,1);
    tmpStr(:) = mkLabel(i);
    mkImages.Labels = tmpStr; 
                            
    countEachLabel(mkImages);
                          
    
    % Split Database into Training & Test Sets in the ration 80% to 20%
    %[mkTrainingSet, mkTestSet] = splitEachLabel(mkImages, 0.8, 'randomize');
    mkTestSet = mkImages;
   
    %% Evaluate the classifier on the test set images and display the confusion matrix    
    [predictedLabelIdx, score] = predict(categoryClassifier, mkTestSet);
       
    
    %% Compute average accuracy
    meanMkAcc = mean(string(categoryClassifier.Labels(predictedLabelIdx)') == string(mkTestSet.Labels));
    
    %details
    %testLabels = categoryClassifier.Labels(predictedLabelIdx)';    
    %M = [testSet.Files, testLabels]';
    %fprintf("%s %s\n", M{:});

    [tn, ~] = size(mkTestSet.Files);
    meanMkConf = zeros(1, nClasses);

    
    maxAccCat = 0;
    maxAcc = 0;
    
    j = 1;   
    for j = 1:nClasses

        tmpStr = strings(tn,1);
        tmpStr(:) = string(categoryClassifier.Labels(j));
    
        meanMkConf(j) = mean(string(categoryClassifier.Labels(predictedLabelIdx)') == tmpStr);

        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end

    %% Results
    fprintf("Makeup Test accuracy: %f \n", meanMkAcc);
    fprintf("Makeup Test Confusion\n");
    fprintf(" %s ",string(categoryClassifier.Labels));
    fprintf("\n %s", mkDataSetFolder(i)); meanMkConf


    
    %% Display poi and matches
    k = 1;
    img1 = readimage(mkTestSet, k);
    % Detect Features for First Image
    img1Pts = detectSURFFeatures(img1);
    [img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts);

    figure; imshow(img1);
    hold on; plot(img1Pts.selectStrongest(50));


    %% find one best match image from the best matching category
    [n,~] = size(allImages.Files);
    in_fl = 0;
    index_pairs_max = 0;
    for m = 1:n
    
        [img2t, info] = readimage(allImages, m);
        if info.Label == maxAccCat
            in_fl = 1;
            img2Pts = detectSURFFeatures(img2t);
            [img2Features,  ~] = extractFeatures(img2t,  img2Pts);

            index_pairs = matchFeatures(img1Features, img2Features);
            [index_pairs_n, ~] = size(index_pairs);
            if index_pairs_n > index_pairs_max
                img2 = img2t;
                index_pairs_max = index_pairs_n;
            end
        else
            if in_fl == 1
                % done with best category
                break
            end
        end
    end

    %% display matching features
    img2Pts = detectSURFFeatures(img2);
    [img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts);

    index_pairs = matchFeatures(img1Features, img2Features);

    img1Matched_pts = img1ValidPts(index_pairs(:,1)).Location;
    img2Matched_pts = img2ValidPts(index_pairs(:,2)).Location;

    figure, 
    showMatchedFeatures(img1, img2, img1Matched_pts,... 
                        img2Matched_pts, 'montage');

end

%dataSetFolder
%fprintf("Training accuracy"); meanAccTrain
%fprintf("Test accuracy");meanAcc

%plotconfusion(categorical(knownLabelIdx), categorical(predictedLabelIdx));