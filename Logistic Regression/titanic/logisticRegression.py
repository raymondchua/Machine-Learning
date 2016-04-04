import numpy as np
class logisticRegression(object):
	"""Logistic Regression

	Parameters
	-----------
	alpha : float
		learning rate (between 0.0 to 1.0)

	n_iter : int
		Passes over the training dataset

	regularization : boolean
		apply regularization to model to prevent
		overfitting

	lambda : float
		regularization value (between 0 to 10)


	Attributes
	-----------
	theta_ : 1d-array
		Weights after fitting

	cost_ : list
		Numer of misclassification in every epoch

	"""

	def __init__(self, alpha = 0.01, n_iter = 10, regularization = False, lambdaVal = 0.0):
		self.alpha = alpha
		self.n_iter = n_iter
		self.regularization = regularization
		self.lambdaVal = lambdaVal

	


	def fit(self, X, y):
		"""Fit training data

		Parameters
		-----------
		X : {array-like}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number
			of samples and n_features is the number of features.

		y : {array-like}, shape = [n_samples]
			Target values.

		Returns
		--------
		self : object
		"""

		
		self.cost_ = []
		m = X.shape[0]
		c =  np.ones((m,1))
		# print(X.shape)
		X = np.insert(X, 0, 1, axis=1)
		self.theta_ = np.zeros(1 + X.shape[1])
		# print(self.theta_.shape)


		"""No regularization"""
		if not self.regularization:
			for _ in range(self.n_iter):
				output = self.hypothesis(X)
				errors =(y-output)
				cost = (1.0/m) * ((-y.T.dot(np.log(output))) - ((1-y).T.dot(np.log(1 - output)))).sum()
				self.cost_.append(cost)
				self.theta_[1:] += self.alpha * (1.0/m) * X.T.dot(errors)
				self.theta_[0] += self.alpha * (1.0/m) * (errors.sum())
		
		else:
			for _ in range(self.n_iter):
				output = self.hypothesis(X)
				errors =(y-output)
				cost = ((1.0/m) * ((-y.T.dot(np.log(output))) - ((1-y).T.dot(np.log(1 - output)))).sum()) + \
				((self.lambdaVal/(2.0*m)) * (np.square(self.theta_)).sum())
				self.cost_.append(cost)
				self.theta_[1:] += self.alpha * (1.0/m) * X.T.dot(errors) + ((self.lambdaVal/m) * self.theta_[1:])
				self.theta_[0] += self.alpha * (1.0/m) * ((errors * X[:,0]).sum())

		

		return self

	def sigmoid(self, h_x):
		"""sigmoid activation function"""
		return 1.0 / (1.0 + np.exp(-h_x))

	def hypothesis(self, X):
		"""Calculate the hypothesis with a sigmoid function

		Return class label where 
		return 1 if theta.T.dot(X) >= 0.0
		return 0 otherwise.
		"""

		temp = np.dot(X, self.theta_[1:]) + self.theta_[0]
		sigmoidTemp = self.sigmoid(temp)
		"""return np.where(sigmoidTemp >= 0.5, 1, 0)"""
		return sigmoidTemp

	def predict(self, X):
		"""Return class label after unit step"""
		m = X.shape[0]
		c =  np.ones((m,1))
		X = np.insert(X, 0, 1, axis=1)
		return np.where(self.hypothesis(X) >= 0.5, 1, 0)

	def cost(self, h_x, y):
		"""return the value from the cost function"""
		m = h_x.shape[0]
		c =  np.ones((m,1))
		h_x = np.insert(h_x, 0, 1, axis=1)
		output = self.hypothesis(h_x)
		errors =(y-output)
		cost = (1.0/m) * ((-y.T.dot(np.log(output))) - ((1-y).T.dot(np.log(1 - output)))).sum()
		return cost




