%Testing code%

spamKaggle = importdata('spam_kaggle.mat');
xtest = spamData.Xtest;
xtest = transpose(xtest);  %columns are observations
xtest = bsxfun(@plus,xtest,0.1);
xtest = log(xtest);
xtest = vertcat(xtest,ones(1,size(xtest,2)));


x = newMat2;  %CHANGE ME FOR DIFFERENT PREPROCESSING
y = double(spamData.ytrain);
numExamples = size(y,1);
numFeatures = 58;
beta = double(zeros(numFeatures,1));  %what we are solving for
lambda = 1;  %regularization parameter
alpha = .00001;  %learning rate
delta = inf;
counter = 5000;
likely = zeros(counter);
count =0;
while count <counter
    muArray = double(zeros(numExamples,1));
    for i=1:numExamples
        muArray(i) = 1.0/(1.0+exp(-transpose(beta)*x(:,i)));
    end
    gradient = x*(muArray-y);
    beta = beta - alpha*(2*lambda*beta + gradient);
    count = count +1;
    likely(count) = lambda*norm(beta)^2 - (transpose(y)*(log(muArray))+ transpose(ones(numExamples,1)-y)*log(ones(numExamples,1)-muArray));
end
    
%Classification
numTests = size(xtest,2);
pred = zeros(numTests,1);
for i = 1:numTests
    pr = 1/(1+exp(-transpose(beta)*xtest(:,i)));
    pred(i) = (pr >= 0.5);
end
csvwrite('submission',pred);

