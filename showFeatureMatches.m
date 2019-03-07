function showFeatureMatches(img1, img2, mkLabel, label, outFolder, accuracy)
%% Display matching features for the test and training images with the maximal number of matches

    if( ~isempty(img1) && ~isempty(img2) )
        img1Pts = detectSURFFeatures(img1);
        [img1Features,  img1ValidPts] = extractFeatures(img1,  img1Pts);
        [img1Pts_n, ~] = size(img1Pts);
     
        img2Pts = detectSURFFeatures(img2);
        [img2Features,  img2ValidPts] = extractFeatures(img2,  img2Pts);
        [img2Pts_n, ~] = size(img2Pts);
    
        index_pairs = matchFeatures(img1Features, img2Features);

        img1Matched_pts = img1ValidPts(index_pairs(:,1)).Location;
        img2Matched_pts = img2ValidPts(index_pairs(:,2)).Location;

        fig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        %subplot(floor(sqrt(n))+1, floor(sqrt(n))+1, i);
        
        showMatchedFeatures(img1, img2, img1Matched_pts,... 
                        img2Matched_pts, 'montage');
        hold on;                
        imgPts_n = max([img1Pts_n, img2Pts_n]);
        plot(img1Pts.selectStrongest(imgPts_n));  
        
        [nPoints, ~] = size(img2Pts.Location);
        [~, xLen] = size(img2);
        offset = zeros(nPoints,1);
        offset(:) = xLen;
        img2Pts.Location(:,1) = img2Pts.Location(:,1) + offset;
        plot(img2Pts.selectStrongest(imgPts_n)); 
        %pause(1);
        
        %
        if ~exist(outFolder, 'dir')
            mkdir(outFolder);
        end
        
        fileName = strcat( outFolder, '/', 'SURF_', mkLabel, '_',...
                    string(label), '_', string(accuracy) );
        saveas(fig, fileName, 'jpeg');
        
        close(fig);
    end
end

