# mnist-numpy
A basic fully connected network implemented purely in NumPy and trained on the MNIST dataset.

## Experiments
The MNIST dataset is split into 50000 train, 10000 validation and 10000 test samples. All splits are normalized using the statistics of the training split (using the global mean and standard deviation, not per pixel).

The network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units. The network is initialized with Xavier-He initialization.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracy on the test set is about 0.97.


## Code structure:
### layers.py
Contains classes that represent layers for different transformations. Each class has a forward and a backward method that define a transformation and its gradient. The class keeps track of the variables defining the transformation and the variables needed to calculate the gradient. The file also contains a class that defines the softmax cross entropy loss.

### network.py
Defines Network, a configurable class representing a sequential neural network with any combination of layers. Network has a train function that performs minibatch SGD.

### main.py
Data loading, training and validation scripts. Running it trains the networks described in experiments. For loading the data it expects two files "data/mnist_train.csv" and "data/mnist_test.csv". These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/. To run use "python3 main.py".
