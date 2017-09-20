"""
    Analyse the nodes of the np_nn 
"""
import numpy as np
from np_nn import Layer, Network
from data import get_test_set, get_training_set

def analyse_layer(current_activation, next_layer: Layer):
    b = np.sum(np.abs(next_layer.bias))
    w = np.sum(np.abs(np.matmul(next_layer.weights, np.diag(current_activation))), 0)
    return w, b

def analyse_layer_multiple(current_activations, next_layer: Layer):
    b = np.sum(np.abs(next_layer.bias))
    w = np.zeros(current_activations[0].shape)
    for a in current_activations:
        w = w + np.sum(np.abs(np.matmul(next_layer.weights, np.diag(a))), 0)
    w = w/len(current_activations)
    return w, b

def get_activations(nn: Network):
    img, lab = get_test_set()
    acts = [[] for _ in range(len(nn.layers))]
    for pa in img:
        pa.shape = 28*28
        for i, layer in enumerate(nn.layers):
            pa, _ = layer.forward()
            acts[i].append(pa)
    return acts




if __name__ == "__main__":
    nn = Network()
    acts = get_activations(nn)
    w, b = analyse_layer_multiple(acts[-2], nn.layers[-1])
    print("Total bias:", b)
    print("Weights:", w)
