"""
    Analyse the nodes of the np_nn
"""
import os
import sys
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from np_nn import Layer, Network, FOLDER, softmax
from data import get_test_set, shuffle_sets

def _forward_backward_analysis(nn: Network, data, label, weights_sum, weights_max, biases, weights_num):
    data.shape = np.prod(data.shape)
    activations = [data]
    for layer in nn.layers:
        a, _ = layer.forward(activations[-1], 0.0)
        activations.append(a)
    tmp = softmax(activations[-1])
    correct = np.argmax(tmp) == label
    activations[-1] = np.zeros_like(activations[-1])
    activations[-1][label] = tmp[label]
    weight = activations[-1]
    weights_sum[-1] += weight
    for i in reversed(range(len(nn.layers))):
        #scale the biases dependent on the ouput node importance
        bias = np.abs(weight*nn.layers[i].bias)
        biases[i+1] += np.sum(bias)
        #scale the previous activations dependent on the ouput node importance
        weight = np.matmul(np.diag(weight), nn.layers[i].weights)
        if i == 0 and  correct:
            weight2 = np.sum(weight, 0)
            weight2.shape = weights_num[label].shape
            weight2 /= np.sum(weight2)+0.0001
            weights_num[label] += weight2
        weight = np.abs(np.matmul(weight, np.diag(activations[i])))
        #save both the sum (later average) and the maximum scaled activation
        weights_max[i] = np.maximum(weights_max[i], np.max(weight, 0))
        weights_sum[i] += np.sum(weight, 0)
        #scale the layer importances with biases and normalise
        bias.shape = bias.shape + (1,)
        division = (weight + np.repeat(bias, weight.shape[1], 1))*(np.sum(weight)+np.sum(bias))+(1e-8) #avoid divide by zero
        weight = np.sum(weight/division, 0)
    return correct

def _combine_analysis(nn: Network, images, labels):
    weights_sum = [np.zeros(np.prod(images[0].shape))]+[np.zeros_like(l.bias) for l in nn.layers]
    weights_max = [np.zeros(np.prod(images[0].shape))]+[np.zeros_like(l.bias) for l in nn.layers]
    biases = np.zeros(len(nn.layers)+1, float)
    weights_num = [np.zeros(images[0].shape[:2]) for _ in range(10)]
    correct = 0
    for img, lab in zip(images, labels):
        if _forward_backward_analysis(nn, img, lab, weights_sum, weights_max, biases, weights_num):
            correct += 1
    print('.', sep='', end='', flush=True)
    return weights_sum, weights_max, biases, correct, weights_num

def backward_analysis(fraction=1):
    print("Loading Data")
    nn = Network()
    images, labels = get_test_set()
    shuffle_sets(images, labels)
    pool_size = 4
    data_size = images.shape[0]//fraction
    size = data_size//(pool_size*2)
    pool = Pool(pool_size)
    data = [(nn, images[i:i+size], labels[i:i+size]) for i in range(0, data_size, size)]
    print("Calculating Node Statistics", end='', flush=True)
    result = pool.starmap(_combine_analysis, data)
    print()
    weights_sum = result[0][0]
    weights_max = result[0][1]
    biases = result[0][2]
    correct = result[0][3]
    weights_num = result[0][4]
    for i in range(1, len(result)):
        for j, _ in enumerate(weights_sum):
            weights_sum[j] += result[i][0][j]
            weights_max[j] = np.maximum(weights_max[j], result[i][0][j])
        biases += result[i][2]
        correct += result[i][3]
        for j, _ in enumerate(weights_num):
            weights_num[j] += result[i][4][j]
    for i in range(len(weights_sum)):
        weights_sum[i] /= data_size
    biases /= data_size
    weights = [np.maximum(a, b) for a, b in zip(weights_max, weights_sum)]
    return weights, biases, float(correct)/float(size*pool_size*2), weights_num

def plot_analysis(weights, biases):
    for i, (w, b) in enumerate(zip(weights, biases)):
        plt.subplot(len(weights), 1, i+1)
        if i == 0:
            plt.title("Input")
            plt.bar(np.arange(w.shape[0]), np.sort(w))
        elif i == len(weights)-1:
            plt.title("Output")
            plt.bar(np.arange(w.shape[0]), w)
            plt.plot([0, w.shape[0]-1], [b, b], 'r')
        else:
            plt.title("Layer %d"%i)
            plt.bar(np.arange(w.shape[0]), np.sort(w))
            plt.plot([0, w.shape[0]-1], [b, b], 'r')
    plt.show()

def plot_images(images, labels):
    v = int(np.ceil(np.sqrt(len(images))))
    h = int(np.ceil(len(images)/v))
    for i, (img, lab) in enumerate(zip(images, labels)):
        plt.subplot(v, h, i+1)
        plt.title(lab)
        plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    res = backward_analysis(10)
    print("Correct: %.3f%%"%(100*res[2]))
    plot_analysis(*res[:2])
    plot_images(res[3], np.arange(10))
