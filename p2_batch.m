%Implementation of batch gradient descent%

x = newMat1;  %CHANGE ME FOR DIFFERENT PREPROCESSING
y = double(spamData.ytrain);
numExamples = size(y,1);
numFeatures = 58;
beta = double(zeros(numFeatures,1));  %what we are solving for
lambda = 1;  %regularization parameter
alpha = .000001;  %learning rate
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
numErrors = 0;
pred = zeros(numExamples,1);
for i = 1:numExamples
    pr = 1/(1+exp(-transpose(beta)*x(:,i)));
    pred(i) = (pr >= 0.5);
   numErrors = numErrors + (pred(i) ~= y(i));
end

