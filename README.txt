PROBLEM 1

Run "p1.m" to get the beta and mu vectors.

PROBLEM 2

First run "p2.m" to load and pre-process the training data.
Run "p2_batch.m" to determine beta using batch gradient descent (change the indicated variable to use different preprocessing methods,
i.e. if you want method 2 change the variable to "newMat2").
Run "p2_stochastic.m" to determine beta using stochastic gradient descent (same deal with the variable).  If you would like to train
with a variable learning rate, uncomment the line under "Variable learning rate".
To perform 10-fold cross validation with a grid search for the optimal learning rate and regularization parameter, run 
"cross_val_grid_search.m".
Finally, to classify the test set, run "tester_new.m".
