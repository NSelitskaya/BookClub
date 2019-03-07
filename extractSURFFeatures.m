function [trainFeatures, trainSets] = extractSURFFeatures(trainingSet)

labels = unique(trainingSet.Labels, 'stable');
%labels = string(categoryClassifier.Labels);
[n, ~] = size(labels); 

labelCounts = table2cell(countEachLabel(trainingSet));
nFiles = max(cell2mat(labelCounts(:,2)));
trainFeatures = cell(nFiles, n);

trainSets = cell(n, 1);

fprintf("Extracting SURF features from images lebeled...\n");

for i=1:n
    fprintf("Label %s\n", labels(i));
    
    %% Find a pair of the correctly identified test and training images with maximum matched features   
    [nTrainFiles, ~] = size(trainingSet.Files);
    tmpStr = strings(nTrainFiles,1);
    tmpStr(:) = string(labels(i));
    
    % Extract file names of the particular category of the train set
    rightTrainFiles = trainingSet.Files( string(trainingSet.Labels) == tmpStr );
    [mFiles, ~] = size(rightTrainFiles);
    rightTrainSet = imageDatastore(string(rightTrainFiles(:,1)));
    rightTrainSet.ReadFcn = trainingSet.ReadFcn;

    trainSets{i} = rightTrainSet;
               
    % Iterate through particular category of the training set
    parfor l=1:mFiles
            
        % Detect features of the training image
        [img2t, ~] = readimage(rightTrainSet, l);
        img2Pts = detectSURFFeatures(img2t);
        [img2Features,  ~] = extractFeatures(img2t,  img2Pts);
                
        trainFeatures{l,i} = img2Features;
        
    end
       
end

end