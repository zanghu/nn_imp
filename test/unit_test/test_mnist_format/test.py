# coding=utf-8
#!/usr/bin/env python
import os
import numpy
import cv2
import time

def txt2img(src_dir, dst_dir):
    """将np.savetxt保存的文件转化为图片"""
    assert os.path.isdir(src_dir)
    if os.path.exists(dst_dir):
        assert os.path.isdir(dst_dir)
    else:
        os.makedirs(dst_dir)
    t0 = time.time()
    for root, dirs, files in os.walk(src_dir):
        for name in files:
            if name.endswith('.txt'):
                src_pth = os.path.join(src_dir, name)
                img = numpy.loadtxt(src_pth, dtype=numpy.uint8)
                dst_pth = os.path.join(dst_dir, name[:-4] + '.bmp')
                cv2.imwrite(dst_pth, img)

    print('all finish, time elapsed: {}s'.format(time.time() - t0))

if __name__ == '__main__':
    txt2img("txt", "img") 
