function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%J
for i=1:m
    temp1=-y(i)*log(sigmoid(X(i,:)*theta));
    temp2=-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    J=J+temp1+temp2;
end
J=J/m;

%grad
vec1=zeros(m,1);
for i=1:m
    vec1(i)=sigmoid(X(i,:)*theta)-y(i);
end
vec1=vec1/m;
grad=X'*vec1;







% =============================================================

end
