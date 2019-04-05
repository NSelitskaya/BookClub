function showSURFFeatureFDMatchesConfusion(bag, trainingSet, trainSets, trainFeatures,...
    trainMetrics, testSet, categoryClassifier, predictedLabelIdx,...
    outFolder, mkTableRow)

% Create a vertical vector of 'objects' containing image file name% its true and predicted label
predictedLabels = categoryClassifier.Labels(predictedLabelIdx)';    
M = [testSet.Files, testSet.Labels, predictedLabels];

%rightTestFiles = M( string(M(:,2)) == string(M(:,3)), : );

% Create a list of labels present in the training set, preserving its occurrence order
labels = unique(trainingSet.Labels, 'stable');
[n, ~] = size(labels); 

% Extract true label of the test subset (from the first full file name)
[tmpStr, ~] = strsplit(testSet.Files{1,1}, '/');
[~, nMatches] = size(tmpStr);
mkLabel = tmpStr{1, nMatches-1};

[nTestFiles, ~] = size(testSet.Files);
readFcn = testSet.ReadFcn;

fprintf("Finding SURF features matches from makeup %s images...\n", mkLabel);

% Iterate through all training set labels
%par
for i=1:n
    %fprintf("Makeup %s images classified as %s\n", mkLabel, labels(i));
    
    % Create a sub-imageDatastore containing test images classified with the given i-th label
    tmpStr = strings(nTestFiles,1);
    tmpStr(:) = string(labels(i));

    rightTestFiles = M( string(M(:,3)) == tmpStr, : );
    [nFiles, ~] = size(rightTestFiles);
    if nFiles == 0
        continue
    end
    rightTestSet = imageDatastore(string(rightTestFiles(:,1)));
    rightTestSet.ReadFcn = readFcn;

    
    %% Find a pair of the test and training images with maximum matched features   
    [mFiles, ~] = size(trainSets{i}.Files);
    
    % Iterate through test set images with the given i-th label
    %index_pairs_max = 0;
    weighted_sum_max = 0;
    img1=[];
    img2=[];
    for k=1:nFiles
        
        % Detect features of the test image
        [img1t, ~] = readimage(rightTestSet, k);   
        [img1Features, ~] = extractFaceSURFFeatures(img1t); 
        %[img1Features, ~] = extractNoFaceSURFFeatures(img1t);
        
        % Iterate through given category in the training set
        for l=1:mFiles
            
            % Get features pre-detected the training image
            img2Features = trainFeatures{l,i};
            img2Metrics = trainMetrics{l,i};
            
            % Find matching features in both images and identify images 
            % with maximal number of them
            if ~isempty(img1Features) && ~isempty(img2Features)            
                index_pairs = matchFeatures(img1Features, img2Features);
                %[index_pairs_n, ~] = size(index_pairs);
                weighted_sum = sum(img2Metrics(index_pairs(:,2)));
                %if index_pairs_n > index_pairs_max
                if weighted_sum > weighted_sum_max
                    img1 = img1t;
                    [img2, ~] = readimage(trainSets{i}, l);
                    weighted_sum_max = weighted_sum;
                    %index_pairs_max = index_pairs_n;
                end
            end
            
        end
        
    end
    
    %% Display matching features for the selected images, and save diagrams into files
    showSURFFeatureFDMatches(bag, img1, img2, mkLabel, labels(i), outFolder, mkTableRow{4+i});
    
    
end

end