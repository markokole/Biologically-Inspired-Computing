"""
The following multi-layer perceptron builds heavily on two sources:
Blog: A Step by Step Backpropagation Example - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
Book: Machine learning, An algorithmic perspective 2nd Edition by Stephen Marsland

At the end is the run section, run it with the default values from the blog. No external files needed to run.
"""
from __future__ import division
import numpy as np
import math

class mlp:
    def __init__(self, n_neurons_input, n_neurons_hidden, n_targets, learning_rate=0.5, beta=1):
        np.set_printoptions(suppress=True) #suppress scientific notation
        self.n_neurons_input = n_neurons_input
        self.n_targets = n_targets # number of target neurons
        self.n_neurons_hidden = n_neurons_hidden # number of neurons in hidden layer

        #intervals for picking random weights for hidden layer
        low_hidden = -1/math.sqrt(self.n_neurons_input)
        high_hidden = 1/math.sqrt(self.n_neurons_input)
        self.weights_hidden = np.random.uniform(low_hidden, high_hidden, size=(self.n_neurons_input, self.n_neurons_hidden))

        #intervals for picking random weights for output layer
        low_output = -1/math.sqrt(self.n_neurons_hidden)
        high_output = 1/math.sqrt(self.n_neurons_hidden)
        self.weights_output =  np.random.uniform(low_output, high_output, size=(self.n_targets, self.n_neurons_hidden))

        self.bias_hidden_weight = np.array([np.random.rand()] * self.n_neurons_hidden)
        self.bias_output_weight = np.array([np.random.rand()] * self.n_targets)
        self.learning_rate = learning_rate
        self.beta = beta #hyperparameter used in sigmoid function

        #prepare the matrices
        self.hidden_layer_activation = np.array([0.] * self.n_neurons_hidden)
        self.hidden_layer_activation_f_output = np.array([0.] * self.n_neurons_hidden)
        self.hidden_layer_error = np.array([0.] * self.n_neurons_hidden)
        self.output_layer_output = np.array([0.] * self.n_targets)
        self.output_layer_activation = np.array([0.] * self.n_targets)
        self.output_layer_error = np.array([0.] * self.n_targets)

        self.all_total_errors = np.array([])

    #activation function - sigmoid is acceptable when having a classification problem
    def sigmoid(self, x, beta):
        return 1 / (1 + math.exp(-x * beta))

    ##calculate hidden layer activations and outputs from activation function
    def forward_hidden_layer(self, inputs):
        hla = np.transpose(np.dot(np.transpose(inputs), self.weights_hidden))
        hlafo = np.array([self.sigmoid(x, self.beta) for x in np.transpose(hla)])
        return hla, hlafo

    ##calculate output layer
    def forward_output_layer(self, sigmoid_from_hidden_layer):
        n_rows = sigmoid_from_hidden_layer.shape[0]
        olo = np.dot(sigmoid_from_hidden_layer.reshape(1,n_rows), np.transpose(self.weights_output)) + self.bias_output_weight
        ola = np.array([self.sigmoid(x, self.beta) for x in np.transpose(olo)])
        return olo, ola

    ##calculate error between actual targets (t) and output targets (o)
    def calculate_total_error(self, t, o):
        total_error = np.sum(1/2 * (t - o) ** 2)
        return total_error

    ##for every output neuron
    def calculate_output_error(self):
        ##compute the error at the output
        self.output_layer_error = (self.output_layer_activation - self.targets) * self.output_layer_activation * (1 - self.output_layer_activation)

    ##calculate the error in the hidden layer
    def backward_hidden_error(self):
        #first part of the equation - the derivative of the activation function - (x*(1-x))
        derivative_part = self.hidden_layer_activation_f_output * (1 - self.hidden_layer_activation_f_output)
        # second part of the equation - sum of product of weights in hidden layer and the error of the output
        n_rows_ole = self.output_layer_error.shape[0]
        n_rows_hla = self.hidden_layer_activation.shape[0]
        #a matrix is returned and the diagonal has the values we are after
        self.hidden_layer_error2 =  np.diagonal(np.transpose(derivative_part) * self.hidden_layer_activation.reshape(n_rows_hla,1) * np.transpose(self.output_layer_error.reshape(n_rows_ole,1)[0]))

    def update_output_layer_weights(self):
        n_rows = self.hidden_layer_activation_f_output.shape[0]
        self.weights_output = self.weights_output - np.transpose(self.learning_rate * self.output_layer_error * np.transpose(self.hidden_layer_activation_f_output.reshape(1,n_rows)))
        self.bias_output_weight = self.bias_output_weight - (self.learning_rate * self.output_layer_error * 1)

    def update_hidden_layer_weights(self):
        n_rows = self.inputs.shape[0]
        self.weights_hidden = self.weights_hidden - (self.learning_rate * self.hidden_layer_error * np.transpose(self.inputs.reshape(1, n_rows)))
        self.bias_hidden_weight = self.bias_hidden_weight - (self.learning_rate * self.hidden_layer_activation * 1)

    ##################
    ##train network  #
    ################
    def train(self, inputs, targets):
        self.inputs = inputs
        self.n_inputs = self.inputs.shape[0]
        self.targets = targets

        ## Forward pass #
        self.hidden_layer_activation, self.hidden_layer_activation_f_output = self.forward_hidden_layer(self.inputs)
        self.output_layer_output, self.output_layer_activation = self.forward_output_layer(self.hidden_layer_activation_f_output)
        self.calculate_total_error(self.targets, self.output_layer_activation)

        ## Backward pass #
        self.calculate_output_error()
        self.backward_hidden_error()
        self.update_output_layer_weights()
        self.update_hidden_layer_weights()
    #end train

    def train_network(self, train, target, n_epoch=10):
        self.n_epoch = n_epoch
        n_rows = train.shape[0]
        for epoch in range(self.n_epoch):
            for i in range(0, n_rows):
                self.train(train[i], target[i])
            print("epoch number:", epoch, ", error=",  self.calculate_total_error(self.targets, self.output_layer_activation))

    def early_stopping(self, train, target, valid, valid_target, n_epoch=10):
        self.n_epoch = n_epoch
        n_rows = train.shape[0]
        previous_valid_error = 100
        message = ""
        for epoch in range(self.n_epoch):
            for i in range(0, n_rows):
                #print("row", i)
                self.train(train[i], target[i])
            #print("early stopping: self.output_layer_activation", self.output_layer_activation.shape, target.shape)
            #print("number of training rows processed:", n_rows)
            #print("VALIDATE!")
            test_error = self.calculate_total_error(self.targets, self.output_layer_activation)
            valid_error = self.forward_pass_with_calculated_error(valid, valid_target)
            #early break is in effect after 10% of the epochs is run
            if (epoch > self.n_epoch/10) and (valid_error > previous_valid_error):
                message = "STOP"
            previous_valid_error = valid_error
            #print("epoch number:", epoch, ", error=", test_error, ", valid error=", valid_error)

            print(epoch, "\t", test_error, "\t",valid_error, "\t", message)
            if message == "STOP":
                break

    def forward_pass_with_calculated_error(self, input_set, target_set):
        #local arrays for activation and sigmoid values
        #hidden_layer_activation = np.array([0.] * self.n_neurons_hidden)
        #hidden_layer_activation_f_output = np.array([0.] * self.n_neurons_hidden)

        output_layer_output = np.array([0.] * self.n_targets)
        output_layer_activation = np.array([0.] * self.n_targets)

        total_error = 0
        n_rows = input_set.shape[0]
        for i in range(n_rows):
            hidden_layer_activation, hidden_layer_activation_f_output = self.forward_hidden_layer(input_set[i])
            output_layer_output, output_layer_activation = self.forward_output_layer(hidden_layer_activation_f_output)

            error = self.calculate_total_error(target_set[i], output_layer_activation)
            total_error = total_error + error

        total_error = total_error/n_rows
        return total_error

    ##vectorized function
    ##method converts the output neuron's value to binary
    def threshold(self, val, threshold_value):
        if val >= threshold_value:
            return 1
        else:
            return 0

    #method delivers final statistics on the model accuracy
    def confusion(self, inputs, targets, threshold_value):
        n_classes = targets.shape[1]
        n_rows = inputs.shape[0]

        vthreshold = np.vectorize(self.threshold)
        percentage = 0

        mtrx = np.zeros((targets.shape[1], targets.shape[1])) #initialize matrix

        for i in range(n_rows):
            hidden_layer_activation, hidden_layer_activation_f_output = self.forward_hidden_layer(inputs[i])
            output_layer_output, output_layer_activation = self.forward_output_layer(hidden_layer_activation_f_output)
            #print("row:", i)
            #print(output_layer_activation)
            binary_output = vthreshold(output_layer_activation, threshold_value) #vectorized function that transforms the scala to binary
            #print("o:", binary_output)
            #print("t:", targets[i])

            t_one_position = np.where(targets[i] == 1)[0] #find position of 1 in array
            o_one_position = np.where(binary_output == 1)[0] #find position of 1 in array
            if o_one_position.shape[0] > 1: #there can be more than one output with value 1, if that is the case, mark it false
                o_one_position = 99
            #print(t_one_position, o_one_position)
            if t_one_position == o_one_position:
                percentage = percentage + 1

            #write matrix
            if (t_one_position < 9 and o_one_position < 9):
                mtrx[t_one_position, o_one_position] = mtrx[t_one_position, o_one_position] + 1

        print("Match coefficient:", round(percentage/n_rows, 4))
        print(mtrx)

##end class neural_network
