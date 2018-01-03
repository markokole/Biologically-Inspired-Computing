from mlp import mlp
import numpy as np

def prepare_data():
    filename = 'movements_day1-3.dat'

    movements = np.loadtxt(filename,delimiter='\t')
    #print(movements)
    #print (movements.shape)

    # Subtract arithmetic mean for each sensor. We only care about how it varies:
    movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

    # Find maximum absolute value:
    imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                              np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                              axis=0 ).max(axis=0)

    # Divide by imax, values should now be between -1,1
    movements[:,:40] = movements[:,:40]/imax[:40]

    # Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
    target = np.zeros((np.shape(movements)[0],8));
    for x in range(1,9):
        indices = np.where(movements[:,40]==x)
        target[indices,x-1] = 1

    # Randomly order the data
    order = list(range(np.shape(movements)[0]))
    np.random.shuffle(order)
    movements = movements[order,:]
    target = target[order,:]

    # Split data into 3 sets

    # Training updates the weights of the network and thus improves the network
    train = movements[::2,0:40]
    train_targets = target[::2]

    # Validation checks how well the network is performing and when to stop
    valid = movements[1::4,0:40]
    valid_targets = target[1::4]

    # Test data is used to evaluate how good the completely trained network is.
    test = movements[3::4,0:40]
    test_targets = target[3::4]

    return train, train_targets, valid, valid_targets, test, test_targets

#end prepare_data

#########
## Run #
#######

input_n_epoch = 10000 # input("Type number of epochs (tested on 10, 100 and 1000): ")
input_n_neuron = 18# input("Type number of neurons in hidden layer (tested on 6, 8 and 12): ")

train, train_targets, valid, valid_targets, test, test_targets = prepare_data()

X = train
target = train_targets
number_of_neurons_in_input = X.shape[1]
number_of_targets = target.shape[1]

number_of_neurons_in_hidden = int(input_n_neuron) #12 #should test with 6, 8, 12
number_of_epochs = int(input_n_epoch) #1000 #10, 100, 1000
net = mlp(number_of_neurons_in_input, number_of_neurons_in_hidden, number_of_targets, beta=5)
net.early_stopping(X, target, valid, valid_targets, number_of_epochs)
threshold_value = 0.5
net.confusion(test, test_targets, threshold_value)
