classdef bagOfFeatures3 < bagOfFeatures
    
    properties(Access = protected)
        goodClusterIdx;
    end
    

    methods (Access = public)
        
        %------------------------------------------------------------------
        function [goodDescriptors, goodMetrics, goodPoints] = extractGoodFeatures(this, img, bad)
                                      
            [descriptors, metrics, locations] = this.Extractor(img);
            points = this.determineExtractionPoints(img);
                                
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
             
            badClusterIdx = ~this.goodClusterIdx;
            
            goodDescriptors = [];
            goodMetrics = [];
            goodLocations = [];
            goodIdx = [];
            
            [n, ~] = size(descriptors);
            if bad == 0
                for i=1:n
                    if matchIndex(i) > 0 && this.goodClusterIdx( matchIndex(i) )
                        goodDescriptors = [goodDescriptors; descriptors(i,:)];
                        goodMetrics = [goodMetrics; metrics(i,:)];
                        goodLocations = [goodLocations; locations(i,:)];
                        goodIdx = [goodIdx, i];
                    end
                end
            else
                for i=1:n
                    if matchIndex(i) > 0 && badClusterIdx( matchIndex(i) )
                        goodDescriptors = [goodDescriptors; descriptors(i,:)];
                        goodMetrics = [goodMetrics; metrics(i,:)];
                        goodLocations = [goodLocations; locations(i,:)];
                        goodIdx = [goodIdx, i];
                    end
                end                
            end
            goodPoints = points(goodIdx);           
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
            if params.UseParallel
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
        % This routine computes a histogram of word occurrences for a given
        % input image.  It turns the input image into a feature vector
        %------------------------------------------------------------------
        function [featureVector, varargout] = encodeSingleImage(this, img, params)
                       
            if nargout == 2                
                [descriptors,~,locations] = this.Extractor(img);
            else
                descriptors = this.Extractor(img);
            end
                                
            opts = getSearchOptions(this);
            
            matchIndex = this.VocabularySearchTree.knnSearch(descriptors, 1, opts); % K = 1
                       
            h = histcounts(single(matchIndex), 1:this.VocabularySize+1);
            
            % Filter out only good cluster center matches
            h = h & this.goodClusterIdx;
            
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
            
            if K < this.VocabularySize
                warning(message('vision:bagOfFeatures:reducingVocabSize', ...
                    K, this.VocabularySize));

                this.VocabularySize = K; 
            end                                              
            
            [clusterCenters, clusterAssignments] = vision.internal.approximateKMeans(descriptors, K, ...
                'Verbose', params.Verbose, 'UseParallel', params.UseParallel);
            
            [m, ~] = size(clusterCenters);
            clusterMemberNum = zeros(m, 1);
            clusterMemberDMean = zeros(m, 1);
            clusterMemberDStd = zeros(m, 1);
            
            for i=1:m
                tmpIdx = zeros(1, numDescriptors);
                tmpIdx(:) = i;
                ithAssignments = clusterAssignments( clusterAssignments == tmpIdx );
                [~, n] = size(ithAssignments);
                clusterMemberNum(i) = n;
                
                ithDescriptors = descriptors( clusterAssignments == tmpIdx, : );
                ithDistDim = ithDescriptors - clusterCenters(i, :);
                ithDist = sqrt(sum(ithDistDim .* ithDistDim, 2));
                clusterMemberDMean(i) = mean(ithDist);
                clusterMemberDStd(i) = std(ithDist);
            end
            
            clusterMembers = [clusterMemberNum, clusterMemberDMean, clusterMemberDStd];
            %fprintf('ClusterCenters & clusterAssignments');
            %array2table(clusterMembers, 'VariableNames',{'Num','Dist','Std'})
            
            [S, L] = bounds(clusterMembers);
            D = L - S;
            %TS = S + D * 0.2;
            %TL = L - D * 0.2;
            TS = mean(clusterMembers) - 1 * std(clusterMembers);
            TL = mean(clusterMembers) + 1 * std(clusterMembers);
            
            %trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1) & clusterMemberDMean > TS(2) & clusterMemberDStd > TS(3), : );
            %trimmedClusterCenters = clusterCenters( clusterMemberDMean > TS(2), : );
            %trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1), : );
            %trimmedClusterCenters = clusterCenters( clusterMemberNum < TL(1) | clusterMemberDMean > TS(2) | clusterMemberDStd > TS(3), : );
            %[n, ~] = size(trimmedClusterCenters);
            
            trimmedClusterCenters = clusterCenters;
            %this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDMean > TS(2) | clusterMemberDStd > TS(3) )';
            %this.goodClusterIdx = ( clusterMemberNum < TL(1) | clusterMemberDStd > TS(3) )';
            this.goodClusterIdx = ( clusterMemberDStd > TS(3) )';
            n = sum(this.goodClusterIdx);
            
            fprintf('bagOfFeatures3: Trimmed clusters number: %d\n', n);
        end
    end
end

