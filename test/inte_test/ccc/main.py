import numpy as np

from layers import Linear, ReLU, SoftmaxCrossEntropyLoss
from network import Network


def main():
    '''
    Trains two networks on the MNIST dataset.
    Both have two hidden ReLU layers with 256 and 128 units
    The fist one has a mean batch normalization layer before every layer
    '''
    np.random.seed(42)
    n_classes = 10
    dim = 784

    inputs, labels = load_normalized_mnist_data()

    # Define network without batch norm
    net = Network(learning_rate = 1e-3)
    net.add_layer(Linear(dim, 256, 'L0_LIN'))
    net.add_layer(ReLU('L0_RELU'))
    net.add_layer(Linear(256, 128, 'L1_LIN'))
    net.add_layer(ReLU('L1_RELU'))
    net.add_layer(Linear(128, n_classes, 'L2_LIN'))
    net.set_loss(SoftmaxCrossEntropyLoss())

    #train_network(net, inputs, labels, 50)
    train_network(net, inputs, labels, 1)
    #test_loss, test_acc = validate_network(net, inputs['test'], labels['test'],
    #                                       batch_size=128)
    #print('Baseline MLP Network without batch normalization:')
    #print('Test loss:', test_loss)
    #print('Test accuracy:', test_acc)
    print('main finish')


def load_normalized_mnist_data():
    '''
    Loads and normalizes the MNIST data. Reads the data from
        data/mnist_train.csv
        data/mnist_test.csv
    These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/
    Returns two dictionaries, input and labels
    Each has keys 'train', 'val', 'test' which map to numpy arrays
    '''
    data = np.loadtxt('/home/zanghu/data_base/mnist/mnist_train.csv', dtype=int, delimiter=',')
    test_data = np.loadtxt('/home/zanghu/data_base/mnist/mnist_test.csv', dtype=int, delimiter=',')

    inputs = dict()
    labels = dict()

    train_data = data[:50000]
    train_inputs = train_data[:, 1:]

    val_data = data[50000:]
    val_inputs = val_data[:, 1:]

    test_inputs = test_data[:, 1:]

    mean = np.mean(train_inputs)
    std = np.std(train_inputs)

    inputs['train'] = (train_inputs - mean)/std
    inputs['val'] = (val_inputs - mean)/std
    inputs['test'] = (test_inputs - mean)/std

    labels['train'] = train_data[:, 0]
    labels['val'] = val_data[:, 0]
    labels['test'] = test_data[:, 0]

    return inputs, labels


def validate_network(network, inputs, labels, batch_size):
    '''
    Calculates loss and accuracy for network when predicting labels from inputs

    Args:
        network (Network): A neural network
        inputs (numpy.ndarray): Inputs to the network
        labels (numpy.ndarray): Labels corresponding to inputs
        batch_size (int): Minibatch size

    Returns:
        avg_loss, accuracy
        avg_loss (float): The average loss per sample using the loss function
                          specified in network
        accuracy (float): (Correct predictions) / (number of samples)
    '''
    n_inputs = inputs.shape[0]

    tot_loss = 0.0
    tot_correct = 0
    start_idx = 0
    while start_idx < n_inputs:
        end_idx = min(start_idx+batch_size, n_inputs)
        mb_inputs = inputs[start_idx:end_idx]
        mb_labels = labels[start_idx:end_idx]

        scores = network.predict(mb_inputs)
        loss, _ = network.loss.get_loss(scores, mb_labels)
        tot_loss += loss * (end_idx-start_idx)
        preds = np.argmax(scores, axis=1)
        tot_correct += np.sum(preds==mb_labels)

        start_idx += batch_size

    avg_loss = tot_loss / n_inputs
    accuracy = tot_correct / n_inputs

    return avg_loss, accuracy


def train_network(network, inputs, labels, n_epochs, batch_size=128):
    '''
    Trains a network for n_epochs

    Args:
        network (Network): The neural network to be trained
        inputs (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        labels (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        n_epochs (int): Specifies number of epochs trained for
        batch_size (int): Number of samples in a minibatch
    '''
    train_inputs = inputs['train']
    train_labels = labels['train']

    n_train = train_inputs.shape[0]

    # Train network
    for epoch in range(n_epochs):
        #order = np.random.permutation(n_train)
        order = np.arange(n_train) # 为了确保和C版对照，取消了训练样本随机乱序
        num_batches = n_train // batch_size
        train_loss = 0

        start_idx = 0
        iter = 0
        while start_idx < n_train:
            end_idx = min(start_idx+batch_size, n_train)
            idxs = order[start_idx:end_idx]
            mb_inputs = train_inputs[idxs]
            mb_labels = train_labels[idxs]

            train_loss += network.train(mb_inputs, mb_labels, epoch, iter)
            start_idx += batch_size
            iter += 1
            if iter >= 2:
                break

        avg_train_loss = train_loss/num_batches

       # avg_val_loss, val_acc = validate_network(network, inputs['val'],
       #                                          labels['val'], batch_size)

        #prnt_tmplt = ('Epoch: {:3}, train loss: {:0.3f}, val loss: ' +
        #             ' {:0.3f}, val acc: {:0.3f}')
        #print(prnt_tmplt.format(epoch, avg_train_loss, avg_val_loss, val_acc))

        prnt_tmplt = ('Epoch: {:3}, train loss: {:0.3f}')
        print(prnt_tmplt.format(epoch, avg_train_loss))

if __name__ == '__main__':
    main()
