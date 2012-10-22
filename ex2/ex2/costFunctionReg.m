function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

for i=1:m
    temp1=-y(i)*log(sigmoid(X(i,:)*theta));
    temp2=-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    J=J+temp1+temp2;
end
J=J/m;
%penal J
penal=lambda*(sum(theta.^2)-theta(1)^2)/(2*m);
J=J+penal;

%grad
vec1=zeros(m,1);
for i=1:m
    vec1(i)=sigmoid(X(i,:)*theta)-y(i);
end
vec1=vec1/m;
grad=X'*vec1;

penal2=lambda*[0 theta(2:length(theta))']/m;

grad=grad+penal2';











% =============================================================

end
