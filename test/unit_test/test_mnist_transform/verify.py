# coding=utf-8
#!/usr/bin/env python

import numpy as np
import os
import time

def load_normalized_mnist_data():
    '''
    Loads and normalizes the MNIST data. Reads the data from
        data/mnist_train.csv
        data/mnist_test.csv
    These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/
    Returns two dictionaries, input and labels
    Each has keys 'train', 'val', 'test' which map to numpy arrays
    '''
    train_data = np.loadtxt('/home/zanghu/data_base/mnist/mnist_train.csv', dtype=int, delimiter=',')
    test_data = np.loadtxt('/home/zanghu/data_base/mnist/mnist_test.csv', dtype=int, delimiter=',')

    inputs = dict()
    labels = dict()

    train_inputs = train_data[:50000, 1:]
    valid_inputs = train_data[50000:, 1:]
    test_inputs = test_data[:, 1:]

    mean = np.mean(train_inputs)
    std = np.std(train_inputs)
    print("mean_py = {}\nstd_py  = {}".format(mean, std))

    inputs['train'] = (train_inputs - mean) / std
    inputs['valid'] = (valid_inputs - mean) / std
    inputs['test'] = (test_inputs - mean) / std

    n_train = train_data.shape[0]
    one_hot = np.zeros((n_train, 10), dtype=np.uint8)
    one_hot[range(n_train), train_data[:, 0]] = 1
    labels['train'] = one_hot[:50000, :]
    labels['valid'] = one_hot[50000:, :]

    n_test = test_data.shape[0]
    one_hot = np.zeros((n_test, 10), dtype=np.uint8)
    one_hot[range(n_test), test_data[:, 0]] = 1
    labels['test'] = one_hot

    return inputs, labels

def compare_mnist(train_data_path, train_label_path, valid_data_path, valid_label_path, test_data_path, test_label_path):
    """"""
    assert os.path.isfile(train_data_path)
    assert os.path.isfile(train_label_path)
    assert os.path.isfile(valid_data_path)
    assert os.path.isfile(valid_label_path)
    assert os.path.isfile(test_data_path)
    assert os.path.isfile(test_label_path)

    t0 = time.time()
    inputs_c  = {}
    inputs_c['train'] = np.loadtxt(train_data_path, dtype=np.float)
    t1 = time.time()
    print("load train_data_c finish, time elapsed: {}s".format(t1 - t0))
    inputs_c['test'] = np.loadtxt(test_data_path, dtype=np.float)
    t2 = time.time()
    print("load test_data_c finish, time elapsed: {}s".format(t2 - t1))
    inputs_c['valid'] = np.loadtxt(valid_data_path, dtype=np.float)
    t3 = time.time()
    print("load valid_data_c finish, time elapsed: {}s".format(t3 - t2))

    labels_c = {}
    labels_c['train'] = np.loadtxt(train_label_path, dtype=np.uint8)
    t4 = time.time()
    print("load train_label_c finish, time elapsed: {}s".format(t4 - t3))
    labels_c['test'] = np.loadtxt(test_label_path, dtype=np.uint8)
    t5 = time.time()
    print("load test_label_c finish, time elapsed: {}s".format(t5 - t4))
    labels_c['valid'] = np.loadtxt(valid_label_path, dtype=np.uint8)
    t6 = time.time()
    print("load valid_label_c finish, time elapsed: {}s".format(t6 - t5))


    inputs_py, labels_py = load_normalized_mnist_data()
    t5 = time.time()
    print("load py finish, time elapsed: {}s".format(t5 - t4))

    # compare
    delta_train_inputs_max = np.max(np.abs(inputs_py['train'] - inputs_c['train']))
    delta_valid_inputs_max = np.max(np.abs(inputs_py['valid'] - inputs_c['valid']))
    delta_test_inputs_max = np.max(np.abs(inputs_py['test'] - inputs_c['test']))

    delta_train_labels_max = np.max(np.abs(labels_py['train'] - labels_c['train']))
    delta_valid_labels_max = np.max(np.abs(labels_py['valid'] - labels_c['valid']))
    delta_test_labels_max = np.max(np.abs(labels_py['test'] - labels_c['test']))
    t6 = time.time()
    print("compare finish, time elapsed: {}s".format(t6 - t5))

    print('delta_train_inputs_max = {}'.format(delta_train_inputs_max))
    print('delta_valid_inputs_max = {}'.format(delta_valid_inputs_max))
    print('delta_test_inputs_max = {}'.format(delta_test_inputs_max))
    print('delta_train_labels_max = {}'.format(delta_train_labels_max))
    print('delta_valid_labels_max = {}'.format(delta_valid_labels_max))
    print('delta_test_labels_max = {}'.format(delta_test_labels_max))

if __name__ == '__main__':
    #load_normalized_mnist_data()
    compare_mnist('txt/mnist_train_images_transformed.txt', 'txt/mnist_train_label_transformed.txt', 'txt/mnist_valid_images_transformed.txt', 'txt/mnist_valid_label_transformed.txt', 'txt/mnist_test_images_transformed.txt', 'txt/mnist_test_label_transformed.txt')
