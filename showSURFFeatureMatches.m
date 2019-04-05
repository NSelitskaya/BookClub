function showSURFFeatureMatches(bag, img1, img2, mkLabel, label, outFolder, accuracy)
%% Display matching features for the test and training images

if( ~isempty(img1) && ~isempty(img2) )
    
    % Length of the first image (to shift coordinates of features in 
    % the second image in motage)
    [~, xLen] = size(img1); 
    
    % Find and extract features
    img1Pts = detectSURFFeatures(img1);
    [img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts, 'Upright', false);
    [img1Pts_n, ~] = size(img1Pts);
     
    img2Pts = detectSURFFeatures(img2);
    [img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts, 'Upright', false);    
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
    plot(img1ValidPts.selectStrongest(imgPts_n));  
    
    img2ValidPts.Location(:,1) = img2ValidPts.Location(:,1) + xLen;
    plot(img2ValidPts.selectStrongest(imgPts_n)); 
    
    pause(1);
        
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

