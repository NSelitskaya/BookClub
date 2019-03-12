function showFeatureMatches(img1, img2, mkLabel, label, outFolder, accuracy)
%% Display matching features for the test and training images

if( ~isempty(img1) && ~isempty(img2) )
    
    % Find and extract features
    img1Pts = detectSURFFeatures(img1);
    [img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts, 'Upright', false);
    [img1Pts_n, ~] = size(img1Pts);
     
    %img2Pts = detectSURFFeatures(img2);
    %[img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts, 'Upright', false);
    [img2Features, ~, img2ValidPts, face2Boxes] = extractFaceSURFFeatures2(img2);
    
    img2 = insertShape(img2, 'Rectangle', face2Boxes, 'LineWidth', 5, 'Color', 'blue' );    
    [img2Pts_n, ~] = size(img2ValidPts);
    
    
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
    [nPoints, ~] = size(img2ValidPts.Location);
    [~, xLen] = size(img1);
    offset = zeros(nPoints,1);
    offset(:) = xLen;
    img2ValidPts.Location(:,1) = img2ValidPts.Location(:,1) + offset;
    plot(img2ValidPts.selectStrongest(imgPts_n)); 
    
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

