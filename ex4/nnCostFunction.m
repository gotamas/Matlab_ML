function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%predict copy
% %X-ben a mint�k soronk�nt
% Z_2=X*Theta1';      %Z_2:5000*25 %mint�k soronk�nt
% m_2 = size(Z_2, 1); %m_2=5000*1 
%  %Z_2=5000*26
% A_2=sigmoid(Z_2); %A_2=5000*26 %mint�k soronk�nt
% A_2 = [ones(m_2, 1) A_2]; 
% Z_3=A_2*Theta2'; %5000x10 %mint�k soronk�nt
% A_3=sigmoid(Z_3); %mint�k soronk�nt
% h_x=A_3;
% 
% 
% Y=zeros(size(h_x));
% for i=1:m
%     Y(i,y(i))=1;
% end
% 
% J=sum(-Y.*log(h_x)-(1-Y).*log(1-h_x))/m;


%probalkozas 2.
J=0;
for i=1:m
    x=X(i,:);

    
    z_2=x*Theta1';      %Z_2:5000*25
    m_2 = size(z_2, 1); %m_2=5000

    a_2=sigmoid(z_2); %A_2=5000*26
    a_2 = [ones(m_2, 1) a_2];
    z_3=a_2*Theta2'; %5000x10 
    a_3=sigmoid(z_3);
    h_x=a_3;
    
    y1=zeros(size(h_x));
    y1(y(i))=y1(y(i))+1;
    
    %cost function
    J=J+sum(-y1.*log(h_x)-(1-y1).*log(1-h_x))/m;
    
    %backwards
   % display(['delta3' size(delta_3); size(delta_2); size(a_3); size(a_2)])
    
    delta_3=a_3-y1;
%    display([size(Theta2'*delta_3'); size(sigmoidGradient(z_2)')])
    
    delta_2=(Theta2'*delta_3');
    delta_2=delta_2(2:end,:).*sigmoidGradient(z_2)';
    
    %delta_2 = delta_2(2:end);
    
 %   display([size(delta_2); size(x); size(Theta1_grad) ])
    Theta1_grad = Theta1_grad + delta_2*x;
   % display([size(delta_3); size(a_2); size(Theta2_grad) ])
    Theta2_grad = Theta2_grad + delta_3'*a_2;
end
    Theta1_grad=Theta1_grad/m;
    Theta2_grad=Theta2_grad/m;


regularizer=lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);
J=J+regularizer;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





%regularization

Theta1_grad=Theta1_grad+lambda*Theta1/m;
Theta1_grad(:,1)=Theta1_grad(:,1)-lambda*Theta1(:,1)/m;

Theta2_grad=Theta2_grad+lambda*Theta2/m;
Theta2_grad(:,1)=Theta2_grad(:,1)-lambda*Theta2(:,1)/m;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
