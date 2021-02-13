# deep_logistic_regression_theano.py

from __future__ import print_function

import gzip
import six.moves.cPickle as pickle
import timeit

import numpy as np 
import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, x, num_in, num_out):
        """
        LogisticRegression model specification in Theano.

        x - Feature vector
        num_in - Dimension of input image (28x28 = 784 for MNIST)
        num_out - Dimension of output (10 for digits 0..9)
        """
        # Initialise the shared weight matrix, 'W'
        self.W = theano.shared(
            value=np.zeros(
                (num_in, num_out), dtype=theano.config.floatX
            ),
            name='W', borrow=True
        )

        # Initialise the shared bias vector, 'b'
        self.b = theano.shared(
            value=np.zeros(
                (num_out,), dtype=theano.config.floatX
            ),
            name='b', borrow=True
        )

        # Symbolic expression for P(Y=k \mid x; \theta)
        self.p_y_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        # Symbolic expression for computing digit prediction
        self.y_pred = T.argmax(self.p_y_x, axis=1)

        # Model parameters
        self.params = [self.W, self.b]

        # Model feature data, x
        self.x = x

    def negative_log_likelihood(self, y):
        """
        Calculate the mean negative log likelihood across
        the N examples in the training data for a minibatch
        """
        return -T.mean(T.log(self.p_y_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # We first check whether the y vector has the same
        # dimension as the y prediction vector
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type)
            )
        # Check if y contains the correct (integer) types
        if y.dtype.startswith('int'):
            # We can use the Theano neq operator to return
            # vector of 1s or 0s, where 1 represents a 
            # classification mistake
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def shared_dataset(data_xy, borrow=True):
    """
    Create Theano shared variables to allow the data to
    be copied to the GPU to avoid performance-degrading
    sequential minibatch data copying.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(
        np.asarray(
            data_x, dtype=theano.config.floatX
        ), borrow=borrow
    )
    shared_y = theano.shared(
        np.asarray(
            data_y, dtype=theano.config.floatX
        ), borrow=borrow
    )
    return shared_x, T.cast(shared_y, 'int32')


def load_mnist_data(filename):
    """
    Load the MNIST gzipped dataset into a
    test, validation and training set via
    the Python pickle library
    """
    # Use the gzip compression and pickle serialisation
    # libraries to open the MNIST images into training,
    # validation and testing sets
    with gzip.open(filename, 'rb') as gzf:
        try:
            train_set, valid_set, test_set = pickle.load(
                gzf, encoding='latin1'
            )
        except:
            train_set, valid_set, test_set = pickle.load(gzf)

    # Use the shared dataset function to create
    # Theano shared variables for GPU data copying
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # Create a list of tuples containing the 
    # feature-response pairs for the training, 
    # validation and test sets respectively
    rval = [
        (train_set_x, train_set_y), 
        (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)
    ]
    return rval


def stoch_grad_desc_train_model(
    filename, gamma=0.13, epochs=1000, B=600
):
    """
    Train the logistic regression model using
    stochastic gradient descent.

    filename - File path of MNIST dataset
    gamma - Step size or "learning rate" for gradient descent
    epochs - Maximum number of epochs to run SGD
    B - The batch size for each minibatch
    """
    # Obtain the correct dataset partitions
    datasets = load_mnist_data(filename)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Calculate the number of minibatches for each of
    # the training, validation and test dataset partitions
    # Note the use of the // operator which is for 
    # integer floor division, e.g.
    # 1.0//2 is equal to 0.0
    # 1//2 is equal to 0
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // B
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // B
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // B

    # BUILD THE MODEL
    # ===============
    print("Building the logistic regression model...")

    # Create symbolic variables for the minibatch data
    index = T.lscalar()  # Integer scalar value
    x = T.matrix('x')  # Feature vectors, i.e. the images
    y = T.ivector('y')  # Vector of integers representing digits

    # Instantiate the logistic regression model and assign the cost
    logreg = LogisticRegression(x=x, num_in=28**2, num_out=10)
    cost = logreg.negative_log_likelihood(y)  # This is what we minimise with SGD

    # Compiles a set of Theano functions for both the testing
    # and validation set that compute the errors on the model
    # in a particular minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=logreg.errors(y),
        givens={
            x: test_set_x[index * B: (index + 1) * B],
            y: test_set_y[index * B: (index + 1) * B]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=logreg.errors(y),
        givens={
            x: valid_set_x[index * B: (index + 1) * B],
            y: valid_set_y[index * B: (index + 1) * B]
        }
    )

    # Use Theano to compute the symbolic gradients of the 
    # cost function (negative log likelihood) with respect to
    # the underlying parameters W and b
    grad_W = T.grad(cost=cost, wrt=logreg.W)
    grad_b = T.grad(cost=cost, wrt=logreg.b)

    # This is the gradient descent step. It specifies a list of
    # tuple pairs, each of which contains a Theano variable and an
    # expression on how to update them on each step.
    updates = [
        (logreg.W, logreg.W - gamma * grad_W),
        (logreg.b, logreg.b - gamma * grad_b)
    ]

    # Similar to the above compiled Theano functions, except that 
    # it is carried out on the training data AND updates the parameters
    # W, b as it evaluates the cost for the particular minibatch
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * B: (index + 1) * B],
            y: train_set_y[index * B: (index + 1) * B]
        }
    )

    # TRAIN THE MODEL
    # ===============
    print("Training the logistic regression model...")

    # Set parameters to stop minibatch early 
    # if performance is good enough
    patience = 5000  # Minimum number of examples to look at
    patience_increase = 2  # Increase by this for new best score
    improvement_threshold = 0.995  # Relative improvement threshold   
    # Train through this number of minibatches before
    # checking performance on the validation set
    validation_frequency = min(n_train_batches, patience // 2)

    # Keep track of the validation loss and test scores
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    # Begin the training loop
    # The outer while loop loops over the number of epochs
    # The inner for loop loops over the minibatches
    finished = False
    cur_epoch = 0
    while (cur_epoch < epochs) and (not finished):
        cur_epoch = cur_epoch + 1
        # Minibatch loop
        for minibatch_index in range(n_train_batches):
            # Calculate the average likelihood for the minibatches
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (cur_epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # If the current iteration has reached the validation
                # frequency then evaluate the validation loss on the
                # validation batches
                validation_losses = [
                    validate_model(i)
                    for i in range(n_valid_batches)
                ]
                this_validation_loss = np.mean(validation_losses)

                # Output current validation results
                print(
                    "Epoch %i, Minibatch %i/%i, Validation Error %f %%" % (
                        cur_epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # If we obtain the best validation score to date
                if this_validation_loss < best_validation_loss:
                    # If the improvement in the loss is within the
                    # improvement threshold, then increase the 
                    # number of iterations ("patience") until the next check
                    if this_validation_loss < best_validation_loss *  \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # Set the best loss to the new current (good) validation loss
                    best_validation_loss = this_validation_loss

                    # Now we calculate the losses on the minibatches 
                    # in the test data set. Our "test_score" is the 
                    # mean of these test losses
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)

                    # Output the current test results (indented)
                    print(
                        (
                            "     Epoch %i, Minibatch %i/%i, Test error of"
                            " best model %f %%"
                        ) % (
                            cur_epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # Serialise the model to disk using pickle
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(logreg, f)
            # If the iterations exceed the current "patience"
            # then we are finished looping for this minibatch
            if iter > patience:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            "Optimization complete with "
            "best validation score of %f %%,"
            "with test performance %f %%"
        ) % (best_validation_loss * 100., test_score * 100.)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec' % (
            cur_epoch, 
            1. * cur_epoch / (end_time - start_time)
        )
    )
    print("The code ran for %.1fs" % (end_time - start_time))


def test_model(filename, num_preds):
    """
    Test the model on unseen MNIST testing data
    """
    # Deserialise the best saved pickled model
    classifier = pickle.load(open('best_model.pkl'))

    # Use Theano to compile a prediction function
    predict_model = theano.function(
        inputs=[classifier.x], 
        outputs=classifier.y_pred
    )

    # Load the MNIST dataset from "filename"
    # and isolate the testing data
    datasets = load_mnist_data(filename)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    # Predict the digits for num_preds images
    preds = predict_model(test_set_x[:num_preds])
    print(
        "Predicted digits for the first %s " \
        "images in test set:" % num_preds
    )
    print(preds)


if __name__ == "__main__":
    # Specify the dataset and the number of 
    # predictions to make on the testing data
    dataset_filename = "mnist.pkl.gz"
    num_preds = 20

    # Train the model via stochastic gradient descent
    stoch_grad_desc_train_model(dataset_filename)

    # Test the logistic regression on unseen test data
    test_model(dataset_filename, num_preds)