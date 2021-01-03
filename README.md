While pursuing the topics of Neural Networks and Deep Learning, these are some of the tutorials which I did, as part of those topics.

**Installation:**

*	Install python>=3.0 or use PyCharm IDE
*	Create a new directory and setup a  conda environment.
*	On the project directory, do **pip install –r requirements.txt**


**1.	Single Layer Neural Networks:**

The neural network model in this tutorial is a single layer of neurons with multiple nodes. 

The activation (transfer) function of each node is assumed to be a hard-limit function.

The test_single_layer_nn.py file includes a very minimal set of unit tests for the single_layer_nn.py part of the tutorial.
To run the tests, use the command **py.test --verbose test_single_layer_nn.py**

**2.	Linear Associator Neural Network:**

In this tutorial a linear associator neural network has been implemented using pseudoinverse rule to calculate the weights and different variations of Hebbian learning rules to train the network. This network is used for classification but it may also be used for regression.

The activation (transfer) function of each node can be linear or hard-limit function. 

The "test_linear_associator.py" file includes a set of unit tests for the linear_associator.py module.
To run the tests, use the command **py.test --verbose test_linear_associator.py**

**3.	Multi-layer Neural Network:**

In this tutorial a multi-layer neural network has been implemented using Tensorflow(without using Keras). 

The activation (transfer) function can be Linear or Relu or Sigmoid.

The test_multinn.py file includes a set of unit tests for the multinn.py module.
To run the tests, use the command **py.test --verbose test_multinn.py**

**4.	Convolutional Neural Network:**

In this tutorial a sequential convolutional neural network (CNN) has been implemented or an existing CNN model has been fine-tuned using Tensorflow.

The activation (transfer) function can be Linear or Relu or Sigmoid.

The test_cnn.py file includes a set of unit tests for the cnn.py module.
To run the tests, use the command **py.test --verbose test_cnn.py**
