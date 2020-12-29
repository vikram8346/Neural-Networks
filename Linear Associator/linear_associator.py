import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.weights = None
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if self.weights.shape[0] != self.number_of_nodes and self.weights.shape[1] != self.input_dimensions:
            return -1
        self.weights = W
        return None

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        # inp_X = np.insert(X, 0, 1, 0)
        if self.transfer_function.lower() == 'linear':
            net = self.weights.dot(X)
            return net
        elif self.transfer_function.lower() == 'hard_limit':
            net = self.weights.dot(X)
            mask = net >= 0
            net[mask] = 1
            net[~mask] = 0
            return net
        # print(net)

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        nW = np.dot(y, np.linalg.pinv(X))
        self.weights = nW

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        r, c = X.shape
        inp_batch = np.split(X, batch_size, 1)
        out_batch = np.split(y, batch_size, 1)
        if learning.lower() == 'delta':
            for epoch in range(num_epochs):
                # for elem in range(r-1//batch_size+1):
                for elem in range(c // batch_size):
                    # s=i*batch_size
                    # e=s+batch_size
                    net_val = self.predict(inp_batch[elem])
                    # net_val=self.predict(X)[s:e] #Transpose?
                    # inp_batch=X[s:e]
                    # out_batch=y[s:e] #Transpose?
                    error = np.subtract(out_batch[elem], net_val)
                    self.weights = np.add(self.weights, alpha * (error.dot(inp_batch[elem].T)))
        elif learning.lower() == 'filtered':
            for epoch in range(num_epochs):
                for elem in range(r - 1 // batch_size + 1):
                    self.weights = np.add((1 - gamma) * self.weights, alpha * (out_batch[elem].dot(inp_batch[elem].T)))
        elif learning.lower() == 'unsupervised_hebb':
            for epoch in range(num_epochs):
                for elem in range(r - 1 // batch_size + 1):
                    net_val = self.predict(inp_batch[elem])
                    self.weights = np.add(self.weights, alpha * (net_val[elem].dot(inp_batch[elem].T)))
        else:
            print("Incorrect Parameter")

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        mse = (np.square(y - self.predict(X))).mean(axis=None)
        return mse
