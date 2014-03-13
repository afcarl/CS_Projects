%Cross_validation
x = newMat3;  %each col is a different observation
y = double(spamData.ytrain);
numExamples = 3450;
numFeatures = 58;

%Dividing randomly into tenths
rand = [1:numExamples];  
rand = rand(randperm(length(rand)));  %permuted list
permData = zeros(numFeatures,numExamples);  %matrix to hold permuted columns
permLabels = zeros(numExamples,1); %vector for permuted labels
for i = 1:length(rand)
    pos = rand(i);
    permData(:,i) = x(:,pos);
    permLabels(i) = y(pos);
end
permData = transpose(permData);

avgErrors = [];
for lambda = [1 10 100]
    for alpha = [0.1,0.01,0.001,0.0001,0.00001]

        numErrors = zeros(10,1);
        for k=0:9
            %Partitioning data
            holdOutData = permData(345*k+1: 345*(k+1),:);
            holdOutLabels = permLabels(345*k+1:345*(k+1));
            modelData = vertcat(permData(1:345*k,:),permData(345*(k+1)+1:3450,:));
            modelLabels = vertcat(permLabels(1:345*k),permLabels(345*(k+1)+1:3450));

            %Determine beta     
            xNew = transpose(modelData);  %obs are in columns
            yNew = modelLabels;
            beta = double(zeros(numFeatures,1));  %what we are solving for
            counter = 2000;
            count =0;
            while count <counter
                muArray = double(zeros(3105,1));
                for i=1:3105
                    muArray(i) = 1.0/(1.0+exp(-transpose(beta)*xNew(:,i)));
                end
                gradient = xNew*(muArray - yNew);
                beta = beta - alpha*(2*lambda*beta + gradient);
                count = count +1;
            end

            %Classification
            pred = zeros(345,1);
            for i = 1:345
                pr = 1/(1+exp(-transpose(beta)*transpose(holdOutData(i,:))));
                pred(i) = (pr >= 0.5);
                numErrors(k+1) = numErrors(k+1) + (pred(i) ~= holdOutLabels(i));
            end
        end
        avgErrors = [avgErrors (mean(numErrors))/345];
    end
end