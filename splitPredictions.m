function [rightSet, wrongSet] = splitPredictions(testSet, categoryClassifier, predictedLabelIdx, score)

    predictedLabels = categoryClassifier.Labels(predictedLabelIdx)';    
    M = [testSet.Files, testSet.Labels, predictedLabels];
    %[n,~] = size(M);
    %dim = ones(n,1);
    %M2 = mat2cell(M, dim);

    rightSet = M( string(M(:,2)) == string(M(:,3)), : );
    wrongSet = M( string(M(:,2)) ~= string(M(:,3)), : );
end