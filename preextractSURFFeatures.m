function [trainFeatures, trainSets, trainBoxes] = preextractSURFFeatures(trainingSet)

persistent faceDetector
if isempty(faceDetector)
    faceDetector = vision.CascadeObjectDetector(); 
end

% Create a list of labels present in the training set, preserving its occurrence order
labels = unique(trainingSet.Labels, 'stable');
[n, ~] = size(labels); 

% Create a cell array of the dimensions capable to hold features % for each image in each category of the training set
labelCounts = table2cell(countEachLabel(trainingSet));
nFiles = max(cell2mat(labelCounts(:,2)));
trainFeatures = cell(nFiles, n);
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
        
        bbox = faceDetector(img2t); % Detect faces
        [m, n] = size(bbox);

        if ~isempty(bbox) && m >= 1 && n == 4  
            img2Pts = detectSURFFeatures(img2t, 'ROI', bbox(1, :));
        else
            [yLen, xLen] = size(img2t);
            bbox = [xLen/2-xLen/6, yLen/2-yLen/6, xLen/3, yLen/3]; % [upper-left x y width hight]
            img2Pts = detectSURFFeatures(img2t, 'ROI', bbox);
        end
        trainBoxes{l,i} = bbox(1);

        [img2Features, ~] = extractFeatures(img2t,  img2Pts, 'Upright', false);
                
        trainFeatures{l,i} = img2Features;
        
    end
       
end

end