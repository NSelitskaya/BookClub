classdef bagOfFeatures4 < bagOfFeatures
    
    properties(Access = protected)
        
        % filtered bag by the calibration bags
        goodClusterIdx;
        badClusterIdx;
        
        % calibration bag clusters data
        clusterMembers;
        
        %SClusterMembers;
        %LClusterMembers;
    end
    

    methods (Access = public)
        
        %------------------------------------------------------------------
        % Constructor
        function obj = bagOfFeatures4(varargin)
            
            obj = obj@bagOfFeatures(varargin{:});

        end % end of Constructor
         
        %------------------------------------------------------------------
        function calculateGoodClusterIdx(this, goodCalibrBag, badCalibrBag)
            
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
                            
            [n, ~] = size(this.Vocabulary);
            
            if ~isempty(goodCalibrBag)
                [m, ~] = size(goodCalibrBag.Vocabulary);                
                vocabDif = zeros(n, m);
                vocabGood = false(n, m);
            
                for i=1:m
                    vocabDifDim = this.Vocabulary - goodCalibrBag.Vocabulary(i, :);
                    vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
                    %vocabGood(:, i) = vocabDif(:, i) < this.clusterMembers(:, 2) + 2*this.clusterMembers(:, 3);
                    vocabGood(:, i) = vocabDif(:, i) < goodCalibrBag.clusterMembers(i, 8);
                end
            
                this.goodClusterIdx = (sum(vocabGood, 2) > 0)';

                k = sum(this.goodClusterIdx);            
                fprintf('bagOfFeatures3: Trimmed by good bag clusters number: %d\n', k);
            end
            
            if ~isempty(badCalibrBag)
                [m, ~] = size(badCalibrBag.Vocabulary);                
                vocabDif = zeros(n, m);
                vocabBad = false(n, m);
            
                for i=1:m
                    vocabDifDim = this.Vocabulary - badCalibrBag.Vocabulary(i, :);
                    vocabDif(:, i) = sqrt(sum(vocabDifDim .* vocabDifDim, 2));
                    %vocabGood(:, i) = vocabDif(:, i) < this.clusterMembers(:, 2) + 2*this.clusterMembers(:, 3);
                    vocabBad(:, i) = vocabDif(:, i) < badCalibrBag.clusterMembers(i, 7);
                end
            
                this.badClusterIdx = (sum(vocabBad, 2) > 0)';

                k = sum(this.badClusterIdx);            
                fprintf('bagOfFeatures3: Trimmed by bad bag clusters number: %d\n', n-k);
            end
           
            fprintf('bagOfFeatures3: Not bad and good intersection %d, union %d\n',...
                sum(this.goodClusterIdx & ~this.badClusterIdx),...
                sum(this.goodClusterIdx | ~this.badClusterIdx));
            
        end
        
        %------------------------------------------------------------------
        function [goodDescriptors, goodMetrics, goodPoints,...
                badDescriptors, badMetrics, badPoints] = extractGoodFeatures(this, img)
                                      
            [descriptors, metrics, locations] = this.Extractor(img);
            points = this.determineExtractionPoints(img);
                                
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            [m, ~] = size(this.Vocabulary);
            goodClusterIdx2 = true(1, m);
            badClusterIdx2 = true(1, m);
            
            if ~isempty(this.goodClusterIdx)
                goodClusterIdx2 = goodClusterIdx2 & this.goodClusterIdx;
            end    
            if ~isempty(this.badClusterIdx)
                goodClusterIdx2 = goodClusterIdx2 & ~this.badClusterIdx;
            end
           
            if ~isempty(this.badClusterIdx)
                badClusterIdx2 = badClusterIdx2 & this.badClusterIdx;
            end    
            if ~isempty(this.goodClusterIdx)
                badClusterIdx2 = badClusterIdx2 & ~this.goodClusterIdx;
            end
            
            
            goodDescriptors = descriptors( goodClusterIdx2(matchIndex), :);
            goodMetrics = metrics( goodClusterIdx2(matchIndex), :);
            goodLocations = locations( goodClusterIdx2(matchIndex), :);
            goodPoints = points( goodClusterIdx2(matchIndex), :);
            
            badDescriptors = descriptors( badClusterIdx2(matchIndex), :);
            badMetrics = metrics( badClusterIdx2(matchIndex), :);
            badLocations = locations( badClusterIdx2(matchIndex), :);
            badPoints = points( badClusterIdx2(matchIndex), :);
                      
        end 
        
        
        
        %------------------------------------------------------------------
        function [goodDescriptors] = filterGoodFeatures(this, descriptors)
                                
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
            
            [m, ~] = size(this.Vocabulary);
            goodClusterIdx2 = true(1, m);
            
            if ~isempty(this.goodClusterIdx)
                goodClusterIdx2 = goodClusterIdx2 & this.goodClusterIdx;
            end    
            if ~isempty(this.badClusterIdx)
                %goodClusterIdx2 = goodClusterIdx2 & ~this.badClusterIdx;
                goodClusterIdx2 = goodClusterIdx2 | ~this.badClusterIdx;
            end           
            
                        
            goodDescriptors = descriptors( goodClusterIdx2(matchIndex), :);           
          
        end         
        
    end      
    
    methods (Hidden, Access = protected)
        
        %------------------------------------------------------------------        
        % Encode a scalar image set. Use parfor if requested.
        %------------------------------------------------------------------
        function [features, varargout] = encodeScalarImageSet(this, imgSet, params)
            
            validateattributes(imgSet,{'imageSet','matlab.io.datastore.ImageDatastore'},{'scalar'},mfilename);
            
            numImages = numel(imgSet.Files);
            
            features = bagOfFeatures.allocateFeatureVector(numImages, this.VocabularySize, params.SparseOutput); 
            words    = bagOfFeatures.allocateVisualWords(numImages);
            
            numVarargout = nargout-1;            
                        
            %DEBUG!!!
            if ~params.UseParallel
                if numVarargout == 1
                    % Invoke 2 output syntax because of parfor limitations
                    % with varargout indexing.                                        
                                                  
                    parfor j = 1:numImages
                        img = imgSet.readimage(j); %#ok<PFBNS>
                        [features(j,:), words(j)]  = this.encodeSingleImage(img, params); %#ok<PFBNS>                        
                    end                
                    
                    varargout{1} = words;
                else                    
                    parfor j = 1:numImages
                        img = imgSet.readimage(j); %#ok<PFBNS>
                        features(j,:)  = this.encodeSingleImage(img, params); %#ok<PFBNS>
                    end
                end
            else % do not use parfor         
                 if numVarargout == 1                                              
                                                  
                    for j = 1:numImages
                        img = imgSet.readimage(j);
                        [features(j,:), words(j)]  = this.encodeSingleImage(img, params);                       
                    end                
                    
                    varargout{1} = words;
                else                    
                    for j = 1:numImages
                        img = imgSet.readimage(j);
                        features(j,:)  = this.encodeSingleImage(img, params);
                    end
                 end
            end
           
        end
        
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
            
            %[goodDescriptors] = this.filterGoodFeatures(descriptors);            
                                            
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
             
                       
            h = histcounts(single(matchIndex), 1:this.VocabularySize+1);
            
            % Filter out only good cluster center matches
            %-------------------------------------------%
            if ~isempty(this.goodClusterIdx)
                h = h & this.goodClusterIdx;                
            end
            
            if ~isempty(this.badClusterIdx)
                h = h & ~this.badClusterIdx;                
            end
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
            
            %--------------------------------------------------------------
            % New from here for bad/good clusters calibration bags
            %--------------------------------------------------------------
            [clusterCenters, clusterAssignments] = vision.internal.approximateKMeans(descriptors, K, ...
                'Verbose', params.Verbose, 'UseParallel', params.UseParallel);
            
            [n, ~] = size(this.Vocabulary);
            this.goodClusterIdx = true(n, 1);
            this.badClusterIdx = true(n, 1);
            
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
                %tmpIdx = zeros(1, numDescriptors);
                %tmpIdx(:) = i;
                %ithAssignments = clusterAssignments( clusterAssignments == tmpIdx );
                ithAssignments = clusterAssignments( clusterAssignments == i );
                [~, n] = size(ithAssignments);
                clusterMemberNum(i) = n/numDescriptors;
                
                %ithDescriptors = descriptors( clusterAssignments == tmpIdx, : );
                ithDescriptors = descriptors( clusterAssignments == i, : );
                ithDistDim = ithDescriptors - clusterCenters(i, :);
                ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
                clusterMemberDMean(i) = mean(ithDist);
                clusterMemberDStd(i) = std(ithDist);
                
                clusterMemberSMean(i) = mean(mean(abs(ithDescriptors), 2));
                clusterMemberSStd(i) = mean(std(abs(ithDescriptors), 0, 2));
                
                clusterMemberDMax(i) = max(ithDist);
                clusterMemberDPrcS(i) = prctile(ithDist, 90); %bad
                clusterMemberDPrcL(i) = prctile(ithDist, 90); %good
            end
            
            this.clusterMembers = [clusterMemberNum, clusterMemberDMean,... 
                clusterMemberDStd, clusterMemberSMean, clusterMemberSStd,...
                clusterMemberDMax, clusterMemberDPrcS, clusterMemberDPrcL];
            
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
            trimmedClusterCenters = clusterCenters;
            
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDMean > TS(2) | clusterMemberDStd > TS(3) )';
            %this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDStd > TS(3) )'; %best
            %%this.goodClusterIdx = ( clusterMemberDStd > TS(3) )';
            %%this.goodClusterIdx = ( clusterMemberDMean > TS(2) )';
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) & clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3) )';
            %%this.goodClusterIdx = ( clusterMemberNum < TL(1) | (clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3)) )';
            %n = sum(this.goodClusterIdx);
            
            %fprintf('bagOfFeatures3: Trimmed clusters number: %d\n', n);
        end
    end
end

