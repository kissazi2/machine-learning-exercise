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

% compute the J
for(i = 1 : m)

   % z =  theta * X(i,:) ;
	z = theta' * X(i,:)';
	h = sigmoid(z);
	logH= log(h);
	logOneSubH = log(1 - h);
	negtiveY = -y(i);
	J = J + (1/m)* sum(negtiveY * logH - ( 1 + negtiveY)* logOneSubH ); 
end

% compute the grad
for(i = 1 : size(theta))
	sumResult = 0;
	for(j = 1:m)
	
		z = theta' * X(j,:)';
		h = sigmoid(z);
		sumResult =sumResult + (h - y(j))*X(j,i);

	end
	grad(i) = sumResult / m;
end

% =============================================================

end
