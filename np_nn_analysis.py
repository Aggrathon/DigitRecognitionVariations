"""
    Analyse the nodes of the np_nn 
"""
import os
import sys
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from np_nn import Layer, Network, FOLDER, softmax
from data import get_test_set

def analyse_layer(current_activation, next_layer: Layer):
    b = np.sum(np.abs(next_layer.bias))
    w = np.sum(np.abs(np.matmul(next_layer.weights, np.diag(current_activation))), 0)
    return w, b

def analyse_layer_multiple(current_activations, next_layer: Layer):
    b = np.sum(np.abs(next_layer.bias))
    w_sum = np.zeros(current_activations[0].shape)
    w_max = np.zeros(current_activations[0].shape)
    for a in current_activations:
        w = np.abs(np.matmul(next_layer.weights, np.diag(a)))
        w_sum = w_sum + np.sum(w, 0)
        w_max = np.maximum(w_max, np.max(w, 0))
    w_sum = w_sum/len(current_activations)
    w_limit = np.maximum(w_sum, w_max)
    return w_sum, w_max, w_limit, b

def plot_weights(weights_sum, weights_max, weights_limit, line):
    plt.bar(np.arange(weights_limit.shape[0]), weights_limit, 0.4)
    plt.bar(np.arange(weights_sum.shape[0])-0.3, weights_sum, 0.2)
    plt.bar(np.arange(weights_max.shape[0])+0.3, weights_max, 0.2)
    plt.plot([0, weights_limit.shape[0]], [line, line])
    plt.show()

def get_activations(nn: Network):
    filename = os.path.join(FOLDER, 'analysis_cache')
    try:
        acts = [np.load(filename+"%d.npy"%i for i in range(len(nn.layers)+1))]
        return acts
    except:
        img, lab = get_test_set()
        acts = [[] for _ in range(len(nn.layers)+1)]
        for pa in img:
            pa.shape = 28*28
            acts[0].append(pa)
            for i, layer in enumerate(nn.layers):
                pa, _ = layer.forward(pa)
                acts[i+1].append(pa)
        try:
            os.makedirs(FOLDER, exist_ok=True)
            for i, a in enumerate(acts):
                np.save(filename+"%d.npy"%i, np.asarray(a))
        except:
            print("Could not cache activations")
            pass
        return acts

def _forward_backward_analysis(nn: Network, data, label, weights, biases):
    data.shape = np.prod(data.shape)
    activations = [data]
    for layer in nn.layers:
        a, _ = layer.forward(activations[-1], 0.0)
        activations.append(a)
    tmp = softmax(activations[-1])
    activations[-1] = np.zeros_like(activations[-1])
    activations[-1][label] = tmp[label]
    weight = activations[-1]
    for i in reversed(range(len(nn.layers))):
        biases[i] += np.sum(np.abs(weight*nn.layers[i].bias))
        weight = np.matmul(np.diag(weight), nn.layers[i].weights)
        weight = np.abs(np.matmul(weight, np.diag(activations[i])))
        weight = np.sum(weight, 0)
        weights[i] += weight

def _combine_analysis(nn: Network, images, labels):
    weights = [np.zeros_like(l.bias) for l in nn.layers]
    biases = np.zeros(len(nn.layers), float)
    for img, lab in zip(images, labels):
        _forward_backward_analysis(nn, img, lab, weights, biases)
    return weights, biases

def backward_analysis():
    pool_size = 4
    nn = Network()
    images, labels = get_test_set()
    pool = Pool(pool_size)
    size = images.shape[0]//pool_size
    data = [(nn, images[i:i+size], labels[i:i+size]) for i in range(0, images.shape[0], size)]
    ws, bs = pool.starmap(_combine_analysis, data)
    weights = ws[0]
    biases = bs[0]
    for i in range(1, len(ws)):
        weights += ws[i]
        biases += bs[i]
    weights /= images.shape[0]
    biases /= images.shape[0]
    for i, (w, b) in enumerate(zip(weights, biases)):
        plt.title("Layer %d"%i)
        plt.bar(np.arange(w.shape[0]), np.sort(w))
        plt.plot([0, w.shape[0]], [b, b])
        plt.show()



if __name__ == "__main__":
    backward_analysis()

