from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # seed random number generator so it generates the same numbers every time it runs
        random.seed(1)

        # model a single neuron with 3 input connections and 1 output connection
        # assign random weights to a 3x1 matrix, with values in range of -1 to 1 & mean 0
        self.synaptic_wieghts = 2 * random.random((3,1)) - 1

    # Sigmoid function describes S shaped curve
    # pass weighted sums of the inputs through this function to normalise them between 1 & 0
    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    # gradient of sigmoid
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass training set through neural network
            output = self.predict(training_set_inputs)

            # calculate error
            error = training_set_outputs - output

            # multiply the error by the input adjust again by the gradient of the sigmoid curve
            adjustments = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust weights
            self.synaptic_wieghts += adjustments

    def predict(self, inputs):
        # pass input through our neural network
        return self.__sigmoid(dot(inputs, self.synaptic_wieghts))

    def think(self, inputs):
        # pass inputs through neural network
        return self.__sigmoid(dot(inputs, self.synaptic_wieghts))


# neural network class

# main function
if __name__ == '__main__':

    # initialise neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights')
    print(neural_network.synaptic_wieghts)

    # training set of 4 examples with 3 inputs & 1 output
    '''
    [0 0 1  [0
     1 1 1   1
     1 0 1   1
     0 1 1]  0]
    '''
    training_set_inputs = array([[0,0,1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    # Classification model: determining to choose either 1 or 0 based on input values
    # train neural network using training set 10,000x with small adjustments each time

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training: ')
    print(neural_network.synaptic_wieghts)

    # test neural network with a new situation
    print('Testing with input [1, 0, 0]: ')
    print(neural_network.think(array([[1, 0, 0]])))
