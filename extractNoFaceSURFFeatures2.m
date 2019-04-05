function [features, featureMetrics, validpoints, noFaceBoxes] = extractNoFaceSURFFeatures2(I)

%
persistent faceDetector
if isempty(faceDetector)
    faceDetector = vision.CascadeObjectDetector(); 
end


bbox = faceDetector(I); % Detect faces
[m, n] = size(bbox);
[yLen, xLen] = size(I);

if isempty(bbox) || m < 1 || n ~= 4      
    bbox = [xLen/2-xLen/6, yLen/2-yLen/6, xLen/3, yLen/3]; % [upper-left x y width hight]
end

bboxN1 = [1, 1, xLen, bbox(1,2)];
bboxN2 = [1, bbox(1,2)+bbox(1,4), xLen, yLen-(bbox(1,2)+bbox(1,4))];
bboxN3 = [1, bbox(1,2), bbox(1,1), bbox(1,4)];
bboxN4 = [bbox(1,1)+bbox(1,3), bbox(1,2), xLen-(bbox(1,1)+bbox(1,3)), bbox(1,4)];
bboxNo = [bboxN1; bboxN2; bboxN3; bboxN4];

points1 = detectSURFFeatures(I, 'ROI', bboxN1);
points2 = detectSURFFeatures(I, 'ROI', bboxN2);
points3 = detectSURFFeatures(I, 'ROI', bboxN3);
points4 = detectSURFFeatures(I, 'ROI', bboxN4);
points = vertcat(points1, points2, points3, points4);

[features, validpoints] = extractFeatures(I, points, 'Upright', false);

noFaceBoxes = bboxNo;

    
featureMetrics = validpoints.Metric;    

 end