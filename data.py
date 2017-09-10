import sys
import numpy as np

def _read_labels(file):
    with open(file, 'rb') as f:
        return np.fromfile(f, dtype=np.int8)[8:]

def _read_images(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.int8)[16:]
        data.shape = (-1, 28, 28)
        return data



if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        print("Number of labels:", _read_labels('t10k-labels.idx1-ubyte').shape)
    if len(sys.argv) == 2 and sys.argv[1] == 'test2':
        print("Number of pixels:", _read_images('t10k-images.idx3-ubyte').shape)
    else:
        print("This script is used to read mnist data files downloaded from 'http://yann.lecun.com/exdb/mnist/'")
