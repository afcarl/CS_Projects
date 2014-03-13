%Implementation of Newton's Method%

x = [0 3 1; 1 3 1; 0 1 1; 1 1 1];
x = transpose(double(x)); %each col is a different observation
y = double([1;1;0;0]);
numExamples = size(y,1);
beta = double([-2; 1; 0]);  %what we are solving for
lambda = 0.07;  %regularization parameter
count = 0;
    while count <2  %run for 2 iterations only
        sum = double(zeros(3,1));
        muArray = double(zeros(numExamples,1));
        for i=1:numExamples
            muArray(i) = 1.0/(1.0+exp(-transpose(beta)*x(:,i)));
            sum = sum + (y(i) + muArray(i)) *x(:,i);
        end
        display(muArray);
        gradient = 2*lambda*beta + sum;
        diagMu = diag(muArray);
        diagLambda = 2*lambda*eye(3);
        hessian = diagLambda - x*diagMu*transpose(x);
        invHessian = hessian\eye(3);
        beta = beta - invHessian*gradient;
        display(beta);
        count = count +1;
    end
       