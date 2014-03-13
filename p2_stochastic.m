%Implementation of stochastic gradient descent%

x = newMat3; %CHANGE ME FOR DIFFERENT PREPROCESSING
y = double(spamData.ytrain);
numExamples = size(y,1);
numFeatures = 58;
beta = double(zeros(numFeatures,1));  %what we are solving for
lambda = 1;%000;  %regularization parameter
alpha = .000001; %learning rate
delta = inf;
counter = 5000;
likely = zeros(numExamples,1);
count =0;
ticker = 1;
while count <counter
    
    %Variable learning rate
    %alpha = (1/(count+1));  %UNCOMMENT ME FOR VARIABLE LEARNING RATE
    
    %Pick random training example
    random = randi([1 numExamples]);
    
    %Updating beta
    mu = 1.0/(1.0+exp(-transpose(beta)*x(:,random)));
    gradient = (mu-y(random))*x(:,random);
    beta = beta - alpha*(2*lambda*beta + gradient);
    
    %Calculate training loss
    muArray = double(zeros(numExamples,1));
    for i = 1:numExamples
        muArray(i) = 1.0/(1.0+exp(-transpose(beta)*x(:,i)));
    end
    count = count +1;
    likely(count) = lambda*norm(beta)^2 - (transpose(y)*(log(muArray))+ transpose(ones(numExamples,1)-y)*log(ones(numExamples,1)-muArray)); 
end
    
%Classification
numErrors = 0;
pred = zeros(numExamples,1);
for i = 1:numExamples
    pr = 1/(1+exp(-transpose(beta)*x(:,i)));
    pred(i) = (pr >= 0.5);
%    if transpose(beta)*x(:,i) > 0
%        pred(i) = 1;  %switched these temporarily
%    else
%        pred(i) = 0;
%    end
   numErrors = numErrors + (pred(i) ~= y(i));
end

