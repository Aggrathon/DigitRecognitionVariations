"""
    A neural network from scratch
"""

import sys
import os
from timeit import default_timer as timer
import numpy as np
from data import get_test_set, get_training_set, shuffle_sets

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


class Layer():

    def __init__(self, input_num, output_num, name, activation=relu, prime=relu_prime):
        self.bias = np.zeros((output_num))
        glorot = np.sqrt(6/(input_num+output_num))
        self.weights = np.random.uniform(-glorot, glorot, (output_num, input_num))
        self.activation = activation
        if activation is not None and prime is None:
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
                        z[i] = 0
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
        self.layers.append(Layer(size, output_size, 'output', None, None))
        self.batches = 0
        try:
            self.batches = np.load(os.path.join(FOLDER, 'meta.npy'))[0]
        except:
            pass
    
    def save(self):
        try:
            for l in self.layers:
                l.save()
            np.save(os.path.join(FOLDER, 'meta.npy'), np.array([self.batches]))
        except:
            pass

    def sgd(self, epochs=20, batch_size=100, learning_rate=0.05, on_epoch=None, on_iter=None):
        """
            Train the network with stochastic gradient descent
        """
        img, lab = get_training_set()
        data_size = img.shape[0]
        img.shape = img.shape[0], np.prod(img.shape[1:])
        if on_iter is None:
            on_iter = Network.get_default_on_iter(data_size//batch_size)
        if on_epoch is None:
            on_epoch = Network.get_default_on_epoch()
        try:
            for i in range(epochs):
                rng = np.random.get_state()
                np.random.shuffle(img)
                np.random.set_state(rng)
                np.random.shuffle(lab)
                for start in range(0, data_size-batch_size+1, batch_size):
                    n_b, n_w, loss = self.backprop(img[start:start+batch_size], lab[start:start+batch_size])
                    lr = learning_rate * 0.9**int(self.batches*8//data_size) / batch_size
                    for layer, b, w in zip(self.layers, n_b, n_w):
                        layer.bias = layer.bias - lr * b
                        layer.weights = layer.weights - lr * w
                    self.batches += 1
                    on_iter(self, i, start//batch_size+1, loss/batch_size)
                on_epoch(self)
        except KeyboardInterrupt:
            print('Aborting...')
            self.evaluate()
        finally:
            self.save()
    
    @classmethod
    def get_default_on_epoch(cls):
        """
            Returns a functions to be used in the sgd when a epoch is finished.
            The function does an evaluation of the network.
        """
        return lambda nn: nn.evaluate()
    @classmethod
    def get_default_on_iter(cls, batches_per_epoch=600):
        """
            Returns a functions to be used in the sgd when a iterations is finished.
            The function prints training stats.
        """
        comb_acc = -1
        time = timer()
        def print_iter(nn: Network, epoch: int, iteration: int, loss: float):
            nonlocal comb_acc
            nonlocal time
            comb_acc = loss if comb_acc == -1 else comb_acc*0.9 + loss*0.1
            if iteration%10 == 0:
                print('[Epoch %d (%d / %d) | %d]   Accuracy: %.3f   (%.2f s)'%(epoch, iteration, batches_per_epoch, nn.batches, 1.0-comb_acc, (timer()-time)/10), end='\r')
                time = timer()
                if iteration%100 == 0:
                    print()
        return print_iter

    def backprop(self, x_list, y_list):
        """
            Calculate backpropagation
        """
        n_b = [(1e-4)*l.bias for l in self.layers]
        n_w = [(1e-4)*l.weights for l in self.layers]
        loss = 0.0
        for x, y in zip(x_list, y_list):
            #forward
            activations = [x]
            zs = []
            for layer in self.layers:
                a, z = layer.forward(activations[-1])
                activations.append(a)
                zs.append(z)
            activations[-1] = softmax(activations[-1])
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

    def evaluate(self, set_size=1.0, print_result=True):
        """
            Evaluate the network
        """
        img, lab = get_test_set()
        img.shape = img.shape[0], np.prod(img.shape[1:])
        correct = 0
        if set_size < 1.0 and set_size >= 0.0:
            shuffle_sets(img, lab)
            size = int(lab.shape[0]*set_size)
        else:
            size = lab.shape[0]
        for i in range(size):
            prev = img[i]
            for layer in self.layers:
                prev, _ = layer.forward(prev, 0)
            if np.argmax(prev) == lab[i]:
                correct += 1
        if print_result:
            print('Evaluation:', correct, '/', lab.shape[0], 'correct')
        return float(correct)/float(lab.shape[0])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'evaluate':
        print(Network().evaluate())
    elif len(sys.argv) == 2 and str.isnumeric(sys.argv[1]):
        net = Network()
        net.sgd(int(sys.argv[1]))
    else:
        net = Network()
        net.sgd()
