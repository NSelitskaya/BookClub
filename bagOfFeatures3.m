classdef bagOfFeatures3 < bagOfFeatures
    
    properties(Access = protected)
        
        % filtered bag by the calibration bags
        goodClusterIdx;
        badClusterIdx;
        
        goodCalibrBag; 
        badCalibrBag;
        
        goodDescriptorIdx;
        badDescriptorIdx;        
        
        origDescriptors;
        origVocabulary;
        
        Verbose;
        UseParallel;
        
        % calibration bag clusters data
        clusterMembers;
        
        %SClusterMembers;
        %LClusterMembers;
    end
    

    methods (Access = public)
        
        %------------------------------------------------------------------
        % Constructor
        function obj = bagOfFeatures3(varargin)
            
            obj = obj@bagOfFeatures(varargin{:});
            
            %obj.goodCalibrBag = t_goodCalibrBag; 
            %obj.badCalibrBag = t_badCalibrBag;

        end % end of Constructor
        
        
                %------------------------------------------------------------------
        function calculateGoodFeatureIdx(this, goodCalibrBag, badCalibrBag)
                        
            opts = getSearchOptions(this);
            
            matchIndex = goodCalibrBag.VocabularySearchTree.knnSearch(this.origDescriptors, 1, opts); % K = 1
            
            ithDistDim = this.origDescriptors - goodCalibrBag.Vocabulary(matchIndex, :);
            ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            
            this.goodDescriptorIdx = (ithDist <= goodCalibrBag.clusterMembers(matchIndex, 6))';

            
            matchIndex = badCalibrBag.VocabularySearchTree.knnSearch(this.origDescriptors, 1, opts); % K = 1
            
            ithDistDim = this.origDescriptors - badCalibrBag.Vocabulary(matchIndex, :);
            ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            
            this.badDescriptorIdx = (ithDist <= badCalibrBag.clusterMembers(matchIndex, 6))';
            
            
                            
            [n, ~] = size(this.origDescriptors);
            
            %if ~isempty(goodCalibrBag)
            %    [m, ~] = size(goodCalibrBag.Vocabulary);                
            %    vocabDif = zeros(n, m);
            %    vocabGood = false(n, m);
            
            %    for i=1:m
            %        vocabDifDim = this.origDescriptors - goodCalibrBag.Vocabulary(i, :);
            %        vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
                    %vocabGood(:, i) = vocabDif(:, i) < goodCalibrBag.clusterMembers(i, 8);
            %        vocabGood(:, i) = vocabDif(:, i) <= goodCalibrBag.clusterMembers(i, 6);
            %    end
            
            %    this.goodDescriptorIdx = (sum(vocabGood, 2) > 0)';

                k = sum(this.goodDescriptorIdx);            
                fprintf('bagOfFeatures3: Trimmed by good bag clusters number: %d\n', k);
            %end
            
            %if ~isempty(badCalibrBag)
            %    [m, ~] = size(badCalibrBag.Vocabulary);                
            %    vocabDif = zeros(n, m);
            %    vocabBad = false(n, m);
            
            %    for i=1:m
            %        vocabDifDim = this.origDescriptors - badCalibrBag.Vocabulary(i, :);
            %        vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
                    %vocabBad(:, i) = vocabDif(:, i) < badCalibrBag.clusterMembers(i, 7);
            %        vocabBad(:, i) = vocabDif(:, i) <= badCalibrBag.clusterMembers(i, 6);
            %    end
            
            %    this.badDescriptorIdx = (sum(vocabBad, 2) > 0)';

                k = sum(this.badDescriptorIdx);            
                fprintf('bagOfFeatures3: Trimmed by bad bag clusters number: %d\n', n-k);
            %end
           
            fprintf('bagOfFeatures3: Not bad and good intersection %d, union %d\n',...
                sum(this.goodDescriptorIdx & ~this.badDescriptorIdx),...
                sum(this.goodDescriptorIdx | ~this.badDescriptorIdx));
            
        end
        
         
        %------------------------------------------------------------------
        %function calculateGoodClusterIdx(this, goodCalibrBag, badCalibrBag)
            
            %minNum = calibrBag.SClusterMembers(1);
            %minDist = calibrBag.SClusterMembers(2);
            %minDStd = calibrBag.SClusterMembers(3);
            %minSym = calibrBag.SClusterMembers(4);
            %minSStd = calibrBag.SClusterMembers(5);
            
            %maxNum = calibrBag.LClusterMembers(1);
            %maxDist = calibrBag.LClusterMembers(2);
            %maxDStd = calibrBag.LClusterMembers(3);
            %maxSym = calibrBag.LClusterMembers(4);
            %maxSStd = calibrBag.LClusterMembers(5);            
            
            %this.goodClusterIdx = ( this.clusterMembers(:,1) >= minNum &... 
            %                        this.clusterMembers(:,1) <= maxNum &...
            %                        this.clusterMembers(:,2) >= minDist &... 
            %                        this.clusterMembers(:,2) <= maxDist &...
            %                        this.clusterMembers(:,3) >= minDStd &... 
            %                        this.clusterMembers(:,3) <= maxDStd &...
            %                        this.clusterMembers(:,4) >= minSym &... 
            %                        this.clusterMembers(:,4) <= maxSym &...
            %                        this.clusterMembers(:,5) >= minSStd &... 
            %                        this.clusterMembers(:,5) <= maxSStd |...
            %                    this.clusterMembers(:,2) == 0 | this.clusterMembers(:,3) == 0 )';
                            
        %    [n, ~] = size(this.origVocabulary);
            
        %    if ~isempty(goodCalibrBag)
        %        [m, ~] = size(goodCalibrBag.Vocabulary);                
        %        vocabDif = zeros(n, m);
        %        vocabGood = false(n, m);
            
        %        for i=1:m
        %            vocabDifDim = this.origVocabulary - goodCalibrBag.Vocabulary(i, :);
        %            vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
        %            vocabGood(:, i) = vocabDif(:, i) < goodCalibrBag.clusterMembers(i, 8);
                    %vocabGood(:, i) = vocabDif(:, i) < goodCalibrBag.clusterMembers(i, 6);
        %        end
            
        %        this.goodClusterIdx = (sum(vocabGood, 2) > 0)';

        %        k = sum(this.goodClusterIdx);            
        %        fprintf('bagOfFeatures3: Trimmed by good bag clusters number: %d\n', k);
        %    end
            
        %    if ~isempty(badCalibrBag)
        %        [m, ~] = size(badCalibrBag.Vocabulary);                
        %        vocabDif = zeros(n, m);
        %        vocabBad = false(n, m);
            
        %        for i=1:m
        %            vocabDifDim = this.origVocabulary - badCalibrBag.Vocabulary(i, :);
        %            vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
        %            vocabBad(:, i) = vocabDif(:, i) < badCalibrBag.clusterMembers(i, 7);
                    %vocabBad(:, i) = vocabDif(:, i) < badCalibrBag.clusterMembers(i, 6);
        %        end
            
        %        this.badClusterIdx = (sum(vocabBad, 2) > 0)';

        %        k = sum(this.badClusterIdx);            
        %        fprintf('bagOfFeatures3: Trimmed by bad bag clusters number: %d\n', n-k);
        %    end
           
        %    fprintf('bagOfFeatures3: Not bad and good intersection %d, union %d\n',...
        %        sum(this.goodClusterIdx & ~this.badClusterIdx),...
        %        sum(this.goodClusterIdx | ~this.badClusterIdx));
            
        %end
        
        
        %------------------------------------------------------------------
        %function [goodDescriptors, goodMetrics, goodPoints,...
        %        badDescriptors, badMetrics, badPoints] = extractGoodFeatures(this, img)
                                      
        %    [descriptors, metrics, locations] = this.Extractor(img);
        %    points = this.determineExtractionPoints(img);
                                
        %    opts = getSearchOptions(this);
            
        %    matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
        %    [m, ~] = size(this.Vocabulary);
        %    goodClusterIdx2 = true(1, m);
        %    badClusterIdx2 = true(1, m);
            
        %    if ~isempty(this.goodClusterIdx)
        %        goodClusterIdx2 = goodClusterIdx2 & this.goodClusterIdx;
        %    end    
        %    if ~isempty(this.badClusterIdx)
        %        goodClusterIdx2 = goodClusterIdx2 & ~this.badClusterIdx;
        %    end
           
        %    if ~isempty(this.badClusterIdx)
        %        badClusterIdx2 = badClusterIdx2 & this.badClusterIdx;
        %    end    
        %    if ~isempty(this.goodClusterIdx)
        %        badClusterIdx2 = badClusterIdx2 & ~this.goodClusterIdx;
        %    end
            
            
        %    goodDescriptors = descriptors( goodClusterIdx2(matchIndex), :);
        %    goodMetrics = metrics( goodClusterIdx2(matchIndex), :);
        %    goodLocations = locations( goodClusterIdx2(matchIndex), :);
        %    goodPoints = points( goodClusterIdx2(matchIndex), :);
            
        %    badDescriptors = descriptors( badClusterIdx2(matchIndex), :);
        %    badMetrics = metrics( badClusterIdx2(matchIndex), :);
        %    badLocations = locations( badClusterIdx2(matchIndex), :);
        %    badPoints = points( badClusterIdx2(matchIndex), :);
                      
        %end         


        %------------------------------------------------------------------
        function [goodDescriptors, goodMetrics, goodPoints,...
                badDescriptors, badMetrics, badPoints] = extractGoodFeaturesByDist(this, img)
                                      
            [descriptors, metrics, locations] = this.Extractor(img);
            points = this.determineExtractionPoints(img);
                                
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            ithDistDim = descriptors - this.Vocabulary(matchIndex, :);
            ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            
            goodIdx = ithDist <= this.clusterMembers(matchIndex, 6);
            goodDescriptors = descriptors( goodIdx, :);
            goodMetrics = metrics( goodIdx, :);
            goodPoints = points( goodIdx, :);
            
            badIdx = ~goodIdx;
            badDescriptors = descriptors( badIdx, :);
            badMetrics = metrics( badIdx, :);
            badPoints = points( badIdx, :);
        end        
        
        
        %------------------------------------------------------------------
        function [goodDescriptors] = filterGoodFeaturesByDist(this, descriptors)
                 
            opts = getSearchOptions(this);
            
            
            
            %matchIndex = this.goodCalibrBag.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            %ithDistDim = descriptors - this.goodCalibrBag.Vocabulary(matchIndex, :);
            %ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            
            %this.goodDescriptorIdx = (ithDist <= this.goodCalibrBag.clusterMembers(matchIndex, 6))';

            
            %matchIndex = this.badCalibrBag.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            %ithDistDim = descriptors - this.badCalibrBag.Vocabulary(matchIndex, :);
            %ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            
            %this.badDescriptorIdx = (ithDist <= this.badCalibrBag.clusterMembers(matchIndex, 6))';
            
            %goodDescriptorIdx2 = this.goodDescriptorIdx | ~this.badDescriptorIdx;
            %goodDescriptors = descriptors(goodDescriptorIdx2, :);
                                
            
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
                               
            ithDistDim = descriptors - this.Vocabulary(matchIndex, :);
            ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
            goodDescriptors = descriptors( ithDist <= this.clusterMembers(matchIndex, 6), :);
            
        end        
              
        
        %------------------------------------------------------------------
        %function [goodDescriptors] = filterGoodFeatures(this, descriptors)
                                                                
        %    opts = getSearchOptions(this);
            
        %    matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            %goodClusterIdx2 = this.goodClusterIdx & ~this.badClusterIdx;
        %    goodClusterIdx2 = this.goodClusterIdx | ~this.badClusterIdx;
            %goodClusterIdx2 = ~this.badClusterIdx;
            %goodClusterIdx2 = this.goodClusterIdx;
            
        %    goodDescriptors = descriptors( goodClusterIdx2(matchIndex), :);
        %end 

        
        %------------------------------------------------------------------
        function trimmedClusterCenters = recreateVocabulary(this, t_goodCalibrBag, t_badCalibrBag)

            this.goodCalibrBag = t_goodCalibrBag; 
            this.badCalibrBag = t_badCalibrBag;

            
            %this.calculateGoodClusterIdx(this.goodCalibrBag, this.badCalibrBag);
            %[goodDescriptors] = this.filterGoodFeatures(this.origDescriptors);

            this.calculateGoodFeatureIdx(this.goodCalibrBag, this.badCalibrBag);
            goodDescriptorIdx2 = this.goodDescriptorIdx | ~this.badDescriptorIdx;
            goodDescriptors = this.origDescriptors(goodDescriptorIdx2, :);
            
            
            numDescriptors = size(goodDescriptors, 1);
            %--------EOCHANGED---------------------------------------------
            
            K = min(numDescriptors, this.VocabularySize); % can't ask for more than you provide
            
            if K == 0
                error(message('vision:bagOfFeatures:zeroVocabSize'))
            end
            
            %if K < this.VocabularySize
            %    warning(message('vision:bagOfFeatures:reducingVocabSize', ...
            %        K, this.VocabularySize));

            %    this.VocabularySize = K; 
            %end                                              
            
            %--------CHANGED-----------------------------------------------
            % New from here for bad/good clusters calibration bags
            %--------------------------------------------------------------
            [trimmedClusterCenters, clusterAssignments] = vision.internal.approximateKMeans(goodDescriptors, K, ...
                'Verbose', this.Verbose, 'UseParallel', this.UseParallel);
            
            %this.origVocabulary = this.Vocabulary;
            [n, ~] = size(this.Vocabulary);
            this.goodClusterIdx = true(n, 1);
            this.badClusterIdx = false(n, 1);
            
            [m, ~] = size(trimmedClusterCenters);
            clusterMemberNum = zeros(m, 1);
            clusterMemberDMean = zeros(m, 1);
            clusterMemberDStd = zeros(m, 1);
            clusterMemberSMean = zeros(m, 1);
            clusterMemberSStd = zeros(m, 1);
            clusterMemberDMax = zeros(m, 1);
            clusterMemberDPrcS = zeros(m, 1);
            clusterMemberDPrcL = zeros(m, 1);
            
            for i=1:m
                ithAssignments = clusterAssignments( clusterAssignments == i );
                [~, n] = size(ithAssignments);
                clusterMemberNum(i) = n/numDescriptors;
                
                ithDescriptors = goodDescriptors( clusterAssignments == i, : );
                ithDistDim = ithDescriptors - trimmedClusterCenters(i, :);
                ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
                clusterMemberDMean(i) = mean(ithDist);
                clusterMemberDStd(i) = std(ithDist);
                
                clusterMemberSMean(i) = mean(mean(abs(ithDescriptors), 2));
                clusterMemberSStd(i) = mean(std(abs(ithDescriptors), 0, 2));
                
                clusterMemberDMax(i) = max(ithDist);
                clusterMemberDPrcS(i) = prctile(ithDist, 90); %bad
                clusterMemberDPrcL(i) = prctile(ithDist, 90); %good 75 85! 90
            end
            
            this.clusterMembers = [clusterMemberNum, clusterMemberDMean,... 
                clusterMemberDStd, clusterMemberSMean, clusterMemberSStd,...
                clusterMemberDMax, clusterMemberDPrcS, clusterMemberDPrcL];
            
            this.Vocabulary = trimmedClusterCenters;
            %--------EOCHANGED---------------------------------------------
            
            this.initializeVocabularySearchTree();
                                        
        end
        
    end
          
    
    methods (Hidden, Access = protected)
        
        %------------------------------------------------------------------        
        % Encode a scalar image set. Use parfor if requested.
        %------------------------------------------------------------------
        %function [features, varargout] = encodeScalarImageSet(this, imgSet, params)
            
        %    validateattributes(imgSet,{'imageSet','matlab.io.datastore.ImageDatastore'},{'scalar'},mfilename);
            
        %    numImages = numel(imgSet.Files);
            
        %    features = bagOfFeatures.allocateFeatureVector(numImages, this.VocabularySize, params.SparseOutput); 
        %    words    = bagOfFeatures.allocateVisualWords(numImages);
            
        %    numVarargout = nargout-1;            
                        
            %DEBUG!!!
        %    if ~params.UseParallel
        %        if numVarargout == 1
                    % Invoke 2 output syntax because of parfor limitations
                    % with varargout indexing.                                        
                                                  
        %            parfor j = 1:numImages
        %                img = imgSet.readimage(j); %#ok<PFBNS>
        %                [features(j,:), words(j)]  = this.encodeSingleImage(img, params); %#ok<PFBNS>                        
        %            end                
                    
        %            varargout{1} = words;
        %        else                    
        %            parfor j = 1:numImages
        %                img = imgSet.readimage(j); %#ok<PFBNS>
        %                features(j,:)  = this.encodeSingleImage(img, params); %#ok<PFBNS>
        %            end
        %        end
        %    else % do not use parfor         
        %         if numVarargout == 1                                              
                                                  
        %            for j = 1:numImages
        %                img = imgSet.readimage(j);
        %                [features(j,:), words(j)]  = this.encodeSingleImage(img, params);                       
        %            end                
                    
        %            varargout{1} = words;
        %        else                    
        %            for j = 1:numImages
        %                img = imgSet.readimage(j);
        %                features(j,:)  = this.encodeSingleImage(img, params);
        %            end
        %         end
        %    end
           
        %end
        
        %------------------------------------------------------------------
        % Copied directly from the standard bagOfFeatures.m with one
        % addition
        %------------------------------------------------------------------
        % This routine computes a histogram of word occurrences for a given
        % input image.  It turns the input image into a feature vector
        %------------------------------------------------------------------
        function [featureVector, varargout] = encodeSingleImage(this, img, params)
                       
            if nargout == 2                
                [descriptors,~,locations] = this.Extractor(img);
            else
                descriptors = this.Extractor(img);
            end
            
            [goodDescriptors] = this.filterGoodFeaturesByDist(descriptors);
                                            
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(goodDescriptors, 1, opts); % K = 1
            
                       
            h = histcounts(single(matchIndex), 1:this.VocabularySize+1);
            
            % Filter out only good cluster center matches
            %-------------------------------------------%
            %if ~isempty(this.goodClusterIdx)
            %    h = h & this.goodClusterIdx;                
            %end
            
            %if ~isempty(this.badClusterIdx)
            %    h = h & ~this.badClusterIdx;                
            %end
            %-------------------------------------------%
            
            featureVector = single(h);
                     
            if strcmpi(params.Normalization,'L2')
                featureVector = featureVector ./ (norm(featureVector,2) + eps('single'));
            end
            
            if params.SparseOutput
                % use sparse storage to reduce memory consumption when
                % featureVector has many zero elements. 
                featureVector = sparse(double(featureVector));
            end
            
            if nargout == 2  
                % optionally return visual words
                varargout{1} = vision.internal.visualWords(matchIndex, locations, this.VocabularySize);                          
            end            
        end        
        
        %------------------------------------------------------------------
        function trimmedClusterCenters = createVocabulary(this, descriptors, varargin)
            
            params = bagOfFeatures.parseCreateVocabularyInputs(varargin{:});
            
            numDescriptors = size(descriptors, 1);
            
            K = min(numDescriptors, this.VocabularySize); % can't ask for more than you provide
            
            if K == 0
                error(message('vision:bagOfFeatures:zeroVocabSize'))
            end
            
            %if K < this.VocabularySize
            %    warning(message('vision:bagOfFeatures:reducingVocabSize', ...
            %        K, this.VocabularySize));

            %    this.VocabularySize = K; 
            %end                                              
            
            %--------CHANGED-----------------------------------------------
            % New from here for bad/good clusters calibration bags
            %--------------------------------------------------------------
            
            %global goodCalibrBag
            %global badCalibrBag
            
            this.origDescriptors = descriptors;

            this.Verbose = params.Verbose;
            this.UseParallel = params.UseParallel;

            
            [clusterCenters, clusterAssignments] = vision.internal.approximateKMeans(descriptors, K, ...
                'Verbose', params.Verbose, 'UseParallel', params.UseParallel);
            
            this.origVocabulary = clusterCenters;
            [n, ~] = size(this.origVocabulary);
            this.goodClusterIdx = true(n, 1);
            this.badClusterIdx = false(n, 1);
            
            [m, ~] = size(clusterCenters);
            clusterMemberNum = zeros(m, 1);
            clusterMemberDMean = zeros(m, 1);
            clusterMemberDStd = zeros(m, 1);
            clusterMemberSMean = zeros(m, 1);
            clusterMemberSStd = zeros(m, 1);
            clusterMemberDMax = zeros(m, 1);
            clusterMemberDPrcS = zeros(m, 1);
            clusterMemberDPrcL = zeros(m, 1);
            
            for i=1:m
                ithAssignments = clusterAssignments( clusterAssignments == i );
                [~, n] = size(ithAssignments);
                clusterMemberNum(i) = n/numDescriptors;
                
                ithDescriptors = descriptors( clusterAssignments == i, : );
                ithDistDim = ithDescriptors - clusterCenters(i, :);
                ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
                clusterMemberDMean(i) = mean(ithDist);
                clusterMemberDStd(i) = std(ithDist);
                
                clusterMemberSMean(i) = mean(mean(abs(ithDescriptors), 2));
                clusterMemberSStd(i) = mean(std(abs(ithDescriptors), 0, 2));
                
                clusterMemberDMax(i) = max(ithDist);
                clusterMemberDPrcS(i) = prctile(ithDist, 90); %bad 90
                clusterMemberDPrcL(i) = prctile(ithDist, 90); %good 99
            end
            
            this.clusterMembers = [clusterMemberNum, clusterMemberDMean,... 
                clusterMemberDStd, clusterMemberSMean, clusterMemberSStd,...
                clusterMemberDMax, clusterMemberDPrcS, clusterMemberDPrcL];
          

            %if ~isempty(goodCalibrBag) && ~isempty(badCalibrBag)
            %    this.goodCalibrBag = goodCalibrBag;
            %    this.badCalibrBag = badCalibrBag;
            %    this.calculateGoodClusterIdx(this.goodCalibrBag, this.badCalibrBag);
            %    %trimmedClusterCenters = clusterCenters(this.goodClusterIdx & ~this.badClusterIdx,:); 
            %    %[K, ~] = size(trimmedClusterCenters);
            %    %this.VocabularySize = K;
            %end
            %else
            trimmedClusterCenters = clusterCenters;
            %end
            
            %fprintf('ClusterCenters & clusterAssignments');
            %array2table(clusterMembers, 'VariableNames',{'Num','Dist','Std'})
            
            %this.SClusterMembers =... 
            %    min(this.clusterMembers( this.clusterMembers(:,2) > 0 & this.clusterMembers(:,3) > 0, : ));
            %this.LClusterMembers =... 
            %    max(this.clusterMembers( this.clusterMembers(:,2) > 0 & this.clusterMembers(:,3) > 0, : ));
            %%this.SClusterMembers =... 
            %%    min(this.clusterMembers);
            %%this.LClusterMembers =... 
            %%    max(this.clusterMembers);
            
            %%D = this.LClusterMembers - this.SClusterMembers;
            %%TS = S + D * 0.2;
            %%TL = L - D * 0.2;
            %TS = mean(this.clusterMembers) - 2 * std(this.clusterMembers);
            %TL = mean(this.clusterMembers) + 2 * std(this.clusterMembers);
            
            %%trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1) & clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3), : );
            %%trimmedClusterCenters = clusterCenters( clusterMemberDMean > TS(2), : );
            %%trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1), : );
            %%trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1) | clusterMemberDMean > TS(2) | clusterMemberDStd > TS(3), : );
            %%[n, ~] = size(trimmedClusterCenters);            
            %trimmedClusterCenters = clusterCenters;
            
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDMean > TS(2) | clusterMemberDStd > TS(3) )';
            %this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDStd > TS(3) )'; %best
            %%this.goodClusterIdx = ( clusterMemberDStd > TS(3) )';
            %%this.goodClusterIdx = ( clusterMemberDMean > TS(2) )';
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) & clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3) )';
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) | (clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3)) )';
            %n = sum(this.goodClusterIdx);
            
            %fprintf('bagOfFeatures3: Trimmed clusters number: %d\n', n);
            %--------EOCHANGED---------------------------------------------            
        end
        
    end
    
end

