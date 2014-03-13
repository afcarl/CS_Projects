spamData = importdata('spam.mat');
xtrain = spamData.Xtrain;
numExamples = size(xtrain,1);  %should be 3450
numFeatures = size(xtrain,2);  %should be 57

%Method 1: mean of 0 std of 1
xtrainTrans = transpose(xtrain);  %columns are now observations
meanVec = mean(xtrain);%mean(xtrainTrans);
stdVec = std(xtrain);%std(xtrainTrans);
newMat1 = bsxfun(@minus,xtrain,meanVec); %bsxfun(@minus,xtrainTrans,meanVec);  %subtract mean from all data points
newMat1 = transpose(bsxfun(@rdivide,newMat1,stdVec));  %divide all data points by std
newMat1 = vertcat(newMat1,ones(1,numExamples));  %add bias feature

%Method 2: transform features
newMat2 = bsxfun(@plus,xtrainTrans,0.1);
newMat2 = log(newMat2);
newMat2 = vertcat(newMat2,ones(1,numExamples));

%Method 3: binarize features
newMat3 = bsxfun(@gt,xtrainTrans,0);
newMat3 = vertcat(newMat3,ones(1,numExamples));