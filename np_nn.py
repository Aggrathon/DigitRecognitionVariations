"""
    A neural network from scratch
"""

import sys
import os
from timeit import default_timer as timer
import numpy as np
from data import get_test_set, get_training_set

FOLDER = 'np_nn'

def relu(x):
    return np.maximum(0.0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    #stabilization
    x = x - np.max(x)
    #softmax
    x = np.exp(x)
    x = x / np.sum(x)
    return x

def softmax_prime(x):
    #Not actually used
    return x


class Layer():

    def __init__(self, input_num, output_num, name, activation=relu, prime=relu_prime):
        self.bias = np.zeros((output_num))
        glorot = np.sqrt(6/(input_num+output_num))
        self.weights = np.random.uniform(-glorot, glorot, (output_num, input_num))
        self.activation = activation
        if prime is None:
            self.prime = lambda z: 1
        else:
            self.prime = prime
        self.name = name
        self.load()

    def forward(self, prev_layer, dropout=0.3):
        z = np.dot(self.weights, prev_layer) + self.bias
        if self.activation is not None:
            result = self.activation(z)
            if dropout > 0:
                scale = 1 / (1-dropout)
                for i in range(result.shape[0]):
                    if np.random.uniform() < dropout:
                        result[i] = 0
                    else:
                        result[i] = result[i]*scale
            return result, z
        else:
            return z, z
    
    def save(self):
        try:
            os.makedirs(FOLDER, exist_ok=True)
            np.save(os.path.join(FOLDER, self.name+'_weights.npy'), self.weights)
            np.save(os.path.join(FOLDER, self.name+'_bias.npy'), self.bias)
        except:
            pass
    
    def load(self):
        try:
            self.weights = np.load(os.path.join(FOLDER, self.name+'_weights.npy'))
            self.bias = np.load(os.path.join(FOLDER, self.name+'_bias.npy'))
        except:
            pass


class Network():

    def __init__(self, input_size=28*28, output_size=10):
        self.layers = []
        size = input_size
        for i, s in enumerate([1024, 512, 128]):
            self.layers.append(Layer(size, s, 'fc%d'%i))
            size = s
        self.layers.append(Layer(size, output_size, 'output', softmax, softmax_prime))

    def sgd(self, epochs=30, batch_size=100, learning_rate=0.05):
        img, lab = get_training_set()
        img.shape = img.shape[0], np.prod(img.shape[1:])
        learning_rate /= batch_size
        try:
            for i in range(epochs):
                rng = np.random.get_state()
                np.random.shuffle(img)
                np.random.set_state(rng)
                np.random.shuffle(lab)
                time = timer()
                loss = 0.0
                for start in range(0, img.shape[0]-batch_size+1, batch_size):
                    n_b, n_w, l = self.backprop(img[start:start+batch_size], lab[start:start+batch_size])
                    for layer, b, w in zip(self.layers, n_b, n_w):
                        layer.bias = layer.bias - learning_rate * b
                        layer.weights = layer.weights - learning_rate * w - (1e-8)*layer.weights
                    loss += l
                    count = start//batch_size+1
                    if count%10 == 0:
                        print('[Epoch %d - %d / %d]   Accuracy: %.3f   (%.2f s)'%(i, count, lab.shape[0]//batch_size, loss/(batch_size*10), (timer()-time)/10))
                        time = timer()
                        loss = 0.0
                learning_rate *= 0.7
                self.evaluate()
            for l in self.layers:
                l.save()
        except KeyboardInterrupt:
            print('Aborting...')
            for l in self.layers:
                l.save()

    def backprop(self, x_list, y_list):
        n_b = [np.zeros_like(l.bias) for l in self.layers]
        n_w = [np.zeros_like(l.weights) for l in self.layers]
        loss = 0.0
        for x, y in zip(x_list, y_list):
            #forward
            activations = [x]
            zs = []
            for layer in self.layers:
                a, z = layer.forward(activations[-1])
                activations.append(a)
                zs.append(z)
            loss += 0.0 if np.argmax(activations[-1]) == y else 1.0
            #backward
            delta = activations[-1]
            delta[y] -= 1
            n_w[-1] += np.outer(delta, activations[-2].T)
            n_b[-1] += delta
            for i in reversed(range(len(self.layers)-1)):
                prime = self.layers[i].prime(zs[i])
                delta = np.dot(self.layers[i+1].weights.T, delta) * prime
                n_b[i] += delta
                n_w[i] += np.outer(delta, activations[i].T)
        return n_b, n_w, loss

    def evaluate(self):
        img, lab = get_test_set()
        img.shape = img.shape[0], np.prod(img.shape[1:])
        correct = 0
        for i in range(lab.shape[0]):
            prev = img[i]
            for layer in self.layers:
                prev, _ = layer.forward(prev, 0)
            if np.argmax(prev) == lab[i]:
                correct += 1
        print('Evaluation:', correct, '/', lab.shape[0], 'correct')


def _test_forward():
    layer = Layer(10, 2)
    inp = np.arange(0.0, 10.0)
    print(layer.forward(inp))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "test1":
        _test_forward()
    else:
        net = Network()
        net.sgd(2)
        net.evaluate()
