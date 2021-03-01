import numpy as np

class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, epoch, iter, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input to the layer with dimensions (batch_size, input_size)

        Returns:
            (numpy.ndarray): Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY, epoch, iter):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

class Linear(Layer):
    def __init__(self, input_dim, output_dim, name):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)

        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))

        self.cache_in = None
        self.name = name

    def forward(self, X, epoch, iter, train=True):
        out = np.matmul(X, self.W) + self.b
        if train:
            self.cache_in = X

        fname_OUT = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'out', '{}x{}'.format(out.shape[0], out.shape[1]))
        np.savetxt(fname_OUT, out, fmt='%.18f')
        fname_W = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'W', '{}x{}'.format(self.W.T.shape[0], self.W.T.shape[1])) # 权重矩阵及其导数矩阵要转置
        np.savetxt(fname_W, self.W.T, fmt='%.18f')
        fname_b = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'b', '{}x{}'.format(self.b.shape[0], self.b.shape[1]))
        np.savetxt(fname_b, self.b, fmt='%.18f')

        return out

    def backward(self, dY, epoch, iter):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        db = np.sum(dY, axis=0, keepdims=True)
        dW = np.matmul(self.cache_in.T, dY)
        dX = np.matmul(dY, self.W.T)
        fname_dY = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'delta', '{}x{}'.format(dY.shape[0], dY.shape[1]))
        np.savetxt(fname_dY, dY, fmt='%.18f')
        fname_dW = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'gW', '{}x{}'.format(dW.T.shape[0], dW.T.shape[1])) # 权重矩阵及其导数矩阵要转置
        np.savetxt(fname_dW, dW.T, fmt='%.18f')
        fname_db = 'txt/epoch_{}_iter_{}_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), self.name, 'gb', '{}x{}'.format(db.shape[0], db.shape[1]))
        np.savetxt(fname_db, db, fmt='%.18f')
        return dX, [(self.W, dW), (self.b, db)]

class ReLU(Layer):
    def __init__(self, name):
        '''
        Represents a rectified linear unit (ReLU)
            ReLU(x) = max(x, 0)
        '''
        self.cache_in = None
        self.name = name

    def forward(self, X, epoch, iter, train=True):
        if train:
            self.cache_in = X
        return np.maximum(X, 0)

    def backward(self, dY, epoch, iter):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (self.cache_in >= 0), []

class Loss(object):
    '''
    Abstract class representing a loss function
    '''
    def get_loss(self):
        raise NotImplementedError('This is an abstract class')

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels, epoch, iter):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores. Shape (batch_size, num_classes)
            labels (numpy.ndarray): True labels represented as ints (eg. 2 represents the third class). Shape (batch_size)

        Returns:
            loss, grad
            loss (float): The average cross entropy between labels and the softmax normalization of scores
            grad (numpy.ndarray): Gradient for scores with respect to the loss. Shape (batch_size, num_classes)
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)

        true_class_scores = scores_norm[np.arange(len(labels)), labels]
        loss = np.mean(-np.log(true_class_scores))
        fname_loss = 'txt/epoch_{}_iter_{}_COST_{}_{}_{}.txt'.format(str(epoch).zfill(3), str(iter).zfill(3), 'CE', 'loss', '1x1')
        np.savetxt(fname_loss, np.asarray([loss]), fmt='%.18f')

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)

        return loss, grad

