function [features, featureMetrics, validpoints] = extractFaceSURFFeatures(I)

%global faceDetector
persistent faceDetector

%features = [];
%featureMetrics = []; 
%validpoints = [];

if isempty(faceDetector)
    faceDetector = vision.CascadeObjectDetector(); 
end

bbox = faceDetector(I); % Detect faces
[m, n] = size(bbox);

if ~isempty(bbox) && m >= 1 && n == 4  
    points = detectSURFFeatures(I, 'ROI', bbox(1, :));
else
    [yLen, xLen] = size(I);
    bbox = [xLen/2-xLen/6, yLen/2-yLen/6, xLen/3, yLen/3]; % [upper-left x y width hight]
    points = detectSURFFeatures(I, 'ROI', bbox);
end

[features, validpoints] = extractFeatures(I, points, 'Upright', false);
    
featureMetrics = validpoints.Metric;    

 end