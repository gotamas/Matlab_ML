function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

params=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];


best=0;
best_sigma=0;
best_C=0;

for i=1:8
    for j=1:8
        sigma1=params(i); C1=params(j);
        model= svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1)); 
        pred = svmPredict(model, Xval);
        temp=mean(double(pred == yval));
        display(['Sigma: ' num2str(sigma1) 'C: ' num2str(C1) 'Pred:' num2str(temp)]);
        if temp>best
            best=temp;
            best_sigma=sigma1;
            best_C=C1;
        end
    end
end

C=best_C;
sigma=best_sigma;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
