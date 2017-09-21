"""
    Analyse the nodes of the np_nn 
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from np_nn import Layer, Network, FOLDER
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


if __name__ == "__main__":
    print("Setting up the network")
    nn = Network()
    print("Calculating activations")
    acts = get_activations(nn)
    print("Calculating node analytics")
    for i in range(len(nn.layers)):
        print("Layer %d"%i)
        w0, w1, w2, b = analyse_layer_multiple(acts[i], nn.layers[i])
        plt.title("Layer %d"%i)
        plot_weights(w0, w1, w2, b)
