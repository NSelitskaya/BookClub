function showFeatureMatches(img1, img2, mkLabel, label, outFolder, accuracy)
%% Display matching features for the test and training images
%global faceDetector
persistent faceDetector
if isempty(faceDetector)
    faceDetector = vision.CascadeObjectDetector(); 
end

if( ~isempty(img1) && ~isempty(img2) )
    
    % Find and extract features
    img1Pts = detectSURFFeatures(img1);
    [img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts, 'Upright', false);
    [img1Pts_n, ~] = size(img1Pts);
     
    %img2Pts = detectSURFFeatures(img2);
    %[img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts, 'Upright', false);
    
    bbox = faceDetector(img2); % Detect faces
    [m, n] = size(bbox);

    if ~isempty(bbox) && m >= 1 && n == 4  
        img2Pts = detectSURFFeatures(img2, 'ROI', bbox(1, :));
    else
        [yLen, xLen] = size(img2);
        bbox = [xLen/2-xLen/6, yLen/2-yLen/6, xLen/3, yLen/3]; % [upper-left x y width hight]
        img2Pts = detectSURFFeatures(img2, 'ROI', bbox);
    end
    [img2Features, img2ValidPts] = extractFeatures(img2,  img2Pts, 'Upright', false);
    
    img2 = insertShape(img2, 'Rectangle', bbox(1, :));    
    [img2Pts_n, ~] = size(img2Pts);
    
    
    % Find matching features and show them on the back-to-back montage
    index_pairs = matchFeatures(img1Features, img2Features);

    img1Matched_pts = img1ValidPts(index_pairs(:,1)).Location;
    img2Matched_pts = img2ValidPts(index_pairs(:,2)).Location;

    fig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    %subplot(floor(sqrt(n))+1, floor(sqrt(n))+1, i);

    showMatchedFeatures(img1, img2, img1Matched_pts,... 
                        img2Matched_pts, 'montage');
                    
    % Display feature areas on top of the first image               
    hold on;                
    imgPts_n = max([img1Pts_n, img2Pts_n]);
    plot(img1Pts.selectStrongest(imgPts_n));  
    
    % Change horisontal coordinates of the second image features by the
    % first image length
    [nPoints, ~] = size(img2Pts.Location);
    [~, xLen] = size(img1);
    offset = zeros(nPoints,1);
    offset(:) = xLen;
    img2Pts.Location(:,1) = img2Pts.Location(:,1) + offset;
    plot(img2Pts.selectStrongest(imgPts_n)); 
    
    %pause(1);
        
    % Create output directory if does not exist and save diagram there
    if ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end
        
    fileName = strcat( outFolder, '/', 'SURF_', mkLabel, '_',...
                    string(label), '_', string(accuracy), '.jpg' );
    saveas(fig, fileName, 'jpeg');
        
    close(fig);
end

end

