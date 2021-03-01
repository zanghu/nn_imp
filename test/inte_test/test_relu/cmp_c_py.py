# coding=utf-8
#!/usr/bin/env python

import numpy as np
import os
import time

def cmp_blob(a, b):
    """"""
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.dtype == float
    return np.max(np.abs(a - b))

def cmp_blobs(py_dir, c_dir, names):
    """比较同名的pyhton数据文件和C数据文件中是内容否一致"""
    assert os.path.isdir(py_dir)
    assert os.path.isdir(c_dir)

    for cnt, name in enumerate(names):
        py_pth = os.path.join(py_dir, name)
        c_pth = os.path.join(c_dir, name)
        #print('py_pth: {}'.format(py_pth))
        #print('c_pth : {}'.format(c_pth))
        assert os.path.isfile(py_pth)
        assert os.path.isfile(c_pth)
        py_blob = np.loadtxt(py_pth)
        c_blob = np.loadtxt(c_pth)
        dist = cmp_blob(py_blob, c_blob)
        print('cnt = {}, {}: {}'.format(cnt, name, dist))

def main():
    """"""
    names = [
             'epoch_000_iter_000_L0_LIN_b_1x256.txt',
             'epoch_000_iter_000_L0_LIN_gb_1x256.txt', 
             'epoch_000_iter_000_L0_LIN_gW_256x784.txt', 
             'epoch_000_iter_000_L0_LIN_out_128x256.txt', 
             'epoch_000_iter_000_L0_LIN_W_256x784.txt', 
             'epoch_000_iter_000_L1_LIN_b_1x128.txt',
             'epoch_000_iter_000_L1_LIN_delta_128x128.txt',
             'epoch_000_iter_000_L1_LIN_gb_1x128.txt',
             'epoch_000_iter_000_L1_LIN_gW_128x256.txt',
             'epoch_000_iter_000_L1_LIN_out_128x128.txt',
             'epoch_000_iter_000_L1_LIN_W_128x256.txt',
             'epoch_000_iter_000_L2_LIN_b_1x10.txt',
             'epoch_000_iter_000_L2_LIN_delta_128x10.txt',
             'epoch_000_iter_000_L2_LIN_gb_1x10.txt',
             'epoch_000_iter_000_L2_LIN_gW_10x128.txt',
             'epoch_000_iter_000_L2_LIN_out_128x10.txt',
             'epoch_000_iter_000_L2_LIN_W_10x128.txt',
             'epoch_000_iter_001_L0_LIN_b_1x256.txt',
             'epoch_000_iter_001_L0_LIN_gb_1x256.txt', 
             'epoch_000_iter_001_L0_LIN_gW_256x784.txt', 
             'epoch_000_iter_001_L0_LIN_out_128x256.txt', 
             'epoch_000_iter_001_L0_LIN_W_256x784.txt', 
             'epoch_000_iter_001_L1_LIN_b_1x128.txt',
             'epoch_000_iter_001_L1_LIN_delta_128x128.txt',
             'epoch_000_iter_001_L1_LIN_gb_1x128.txt',
             'epoch_000_iter_001_L1_LIN_gW_128x256.txt',
             'epoch_000_iter_001_L1_LIN_out_128x128.txt',
             'epoch_000_iter_001_L1_LIN_W_128x256.txt',
             'epoch_000_iter_001_L2_LIN_b_1x10.txt',
             'epoch_000_iter_001_L2_LIN_delta_128x10.txt',
             'epoch_000_iter_001_L2_LIN_gb_1x10.txt',
             'epoch_000_iter_001_L2_LIN_gW_10x128.txt',
             'epoch_000_iter_001_L2_LIN_out_128x10.txt',
             'epoch_000_iter_001_L2_LIN_W_10x128.txt'
            ]

    project_dir = '../../..'
    py_dir = os.path.join(project_dir, 'test/inte_test/ccc/txt')
    c_dir = os.path.join(project_dir, 'test/inte_test/test_relu/txt')
    cmp_blobs(py_dir, c_dir, names)

if __name__ == '__main__':
    main()
