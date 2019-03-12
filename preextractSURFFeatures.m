function [trainSets,trainFeatures, trainMetrics, trainBoxes] =...
    preextractSURFFeatures(trainingSet)

% Create a list of labels present in the training set, preserving its occurrence order
labels = unique(trainingSet.Labels, 'stable');
[n, ~] = size(labels); 

% Create a cell array of the dimensions capable to hold features % for each image in each category of the training set
labelCounts = table2cell(countEachLabel(trainingSet));
nFiles = max(cell2mat(labelCounts(:,2)));
trainFeatures = cell(nFiles, n);
trainMetrics = cell(nFiles, n);
trainBoxes = cell(nFiles, n);

trainSets = cell(n, 1);

fprintf("Extracting SURF features from images lebeled...\n");

for i=1:n
    fprintf("Label %s\n", labels(i));
    
    % Extract file names of the particular category of the train set    
    % and create sub-imageDatastores for them  
    [nTrainFiles, ~] = size(trainingSet.Files);
    tmpStr = strings(nTrainFiles,1);
    tmpStr(:) = string(labels(i));
    
    rightTrainFiles = trainingSet.Files( string(trainingSet.Labels) == tmpStr );
    [mFiles, ~] = size(rightTrainFiles);
    rightTrainSet = imageDatastore(string(rightTrainFiles(:,1)));
    rightTrainSet.ReadFcn = trainingSet.ReadFcn;

    trainSets{i} = rightTrainSet;
               
    % Iterate through particular category of the training set
    parfor l=1:mFiles
            
        % Detect features of the training image
        [img2t, ~] = readimage(rightTrainSet, l);
        
        %img2Pts = detectSURFFeatures(img2t);
        %[img2Features,  ~] = extractFeatures(img2t,  img2Pts);        
        [img2Features, face2Metrics, ~, face2Boxes] = extractFaceSURFFeatures2(img2t);
                        
        trainFeatures{l,i} = img2Features;
        trainMetrics{l,i} = face2Metrics; 
        trainBoxes{l,i} = face2Boxes; 
    end
       
end

end