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
from np_utils import np_onehot

def _fb_analysis(nn: Network, data, label, weights_sum, weights_max, biases, weights_num):
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
        if i == 0:
            weight2 = np.sum(np.abs(weight), 0)
            weight2.shape = weights_num[label].shape
            weight2 /= np.sum(weight2)+0.0001
            weights_num[-1] += weight2
            if correct:
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

def _combine_fb_analysis(nn: Network, images, labels):
    weights_sum = [np.zeros(np.prod(images[0].shape))]+[np.zeros_like(l.bias) for l in nn.layers]
    weights_max = [np.zeros(np.prod(images[0].shape))]+[np.zeros_like(l.bias) for l in nn.layers]
    biases = np.zeros(len(nn.layers)+1, float)
    weights_num = [np.zeros(images[0].shape[:2]) for _ in range(11)]
    correct = 0
    for img, lab in zip(images, labels):
        if _fb_analysis(nn, img, lab, weights_sum, weights_max, biases, weights_num):
            correct += 1
    print('.', sep='', end='', flush=True)
    return weights_sum, weights_max, biases, correct, weights_num

def forward_backward_analysis(nn: Network=None, fraction=1):
    if nn is None:
        nn = Network()
    print("Loading Data")
    images, labels = get_test_set()
    shuffle_sets(images, labels)
    pool_size = 4
    data_size = images.shape[0]//fraction
    splits = int(np.sqrt(np.maximum(10-fraction, 0))+1)
    size = data_size//(pool_size*splits)
    pool = Pool(pool_size)
    data = [(nn, images[i:i+size], labels[i:i+size]) for i in range(0, data_size, size)]
    print("Calculating Node Statistics", end='', flush=True)
    result = pool.starmap(_combine_fb_analysis, data)
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
    return weights, biases, float(correct)/float(size*pool_size*splits), weights_num


def backward_analysis(nn: Network=None):
    if nn is None:
        nn = Network()
    cases = [np_onehot(10, i, float) for i in range(10)] + [np.ones(10, float)]
    weights_sum = [np.zeros(nn.layers[0].weights.shape[1])]+[np.zeros_like(l.bias) for l in nn.layers]
    weights_max = [np.zeros(nn.layers[0].weights.shape[1])]+[np.zeros_like(l.bias) for l in nn.layers]
    biases = np.zeros(len(nn.layers)+1, float)
    weights_num = [np.zeros((28, 28)) for _ in range(11)]
    for case, output in enumerate(cases):
        weight = output
        for i in reversed(range(len(nn.layers))):
            #scale the biases dependent on the ouput node importance
            biases[i+1] += np.sum(np.abs(weight*nn.layers[i].bias))
            #scale the previous layer nodes dependent on the ouput node importance
            weight = np.abs(np.matmul(np.diag(weight), nn.layers[i].weights))
            #save both the sum (later average) and the maximum importances
            weights_max[i] = np.maximum(weights_max[i], np.max(weight, 0))
            weight = np.sum(weight, 0)
            weights_sum[i] += weight
            #normalise the layer importances
            weight = weight/(np.sum(weight)+(1e-8))
            #save pixel activations
            if i == 0:
                weight.shape = weights_num[case].shape
                weights_num[-1] += weight
                weights_num[case] += weight
    for i in range(len(weights_sum)):
        weights_sum[i] /= len(cases)
    biases /= len(cases)
    weights = [np.maximum(a, b) for a, b in zip(weights_max, weights_sum)]
    weights[-1] = np.sum(cases, 0) / len(cases)
    return weights, biases, 0.0, weights_num

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
        plt.subplot(h, v, i+1)
        plt.title(lab)
        plt.imshow(img, cmap='gray')
    plt.show()


def reactivate_dead_nodes(nn, weights=None, biases=None):
    if biases is None or weights is None:
        weights, biases = forward_backward_analysis(nn)[:2]
    reac = 0
    means = [np.mean(np.abs(l.weights)) for l in nn.layers]
    for i, layer in enumerate(weights[1:]):
        b0 = biases[i]*0.4
        b1 = np.max(layer)*0.1
        b2 = np.mean(layer)-np.std(layer)
        b3 = np.mean(layer)*0.2
        b = max(1e-4, min(b0, b1, b2, b3))
        """
        if b == b0:
            print("bias")
        elif b == b1:
            print("max")
        elif b == b2:
            print("mean-std")
        else:
            print("min")
        """
        for j, w in enumerate(layer):
            if w < b:
                reac += 1
                lw = nn.layers[i].weights
                for k in range(lw.shape[1]):
                    lw[j, k] = np.random.uniform(-means[i], means[i])*3
                if i+1 < len(nn.layers):
                    lw = nn.layers[i+1].weights
                    for k in range(lw.shape[0]):
                        lw[k, j] = np.random.uniform(-means[i+1], means[i+1])*2
    print("%d nodes reactivated"%reac)

def num_dead_nodes(nn: Network, weights=None, biases=None):
    if biases is None or weights is None:
        weights, biases = forward_backward_analysis(nn)[:2]
    reac = 0
    for i, layer in enumerate(weights[1:]):
        b = max(1e-4, min(biases[i]*0.4, np.max(layer)*0.1, np.mean(layer)-np.std(layer), np.mean(layer)*0.2))
        for w in layer:
            if w < b:
                reac += 1
    print("%d dead nodes"%reac)
    return reac

def _cl_learn_prog(nn: Network, epochs: int=1):
    progression = []
    on_iter_orig = Network.get_default_on_iter()
    def on_iter(nn: Network, epoch: int, iteration: int, loss: float):
        nonlocal progression
        nonlocal on_iter_orig
        nonlocal epochs
        on_iter_orig(nn, epoch, iteration, loss)
        if iteration%100 == 0:
            progression.append(nn.evaluate(set_size=1.0 if epochs < 3 else 0.5))
    nn.sgd(epochs, on_iter=on_iter, on_epoch=lambda nn: None)
    return progression

def compare_learning(epochs=2):
    e1 = (epochs*2)//3
    e2 = epochs-e1
    nn1 = Network()
    print("Beginning with some normal epochs for both options")
    prog1 = [nn1.evaluate()] + _cl_learn_prog(nn1, e1)
    nn1.save()
    nn2 = Network()
    print("First: a normal epoch")
    prog2 = prog1 + _cl_learn_prog(nn1, e2)
    print("Second: a modified epoch")
    reactivate_dead_nodes(nn2)
    prog_re = nn2.evaluate()
    prog3 = prog1 + _cl_learn_prog(nn2, e2)
    for i, p in enumerate(prog1):
        prog3[i] = p-(1e-5)
    prog3[len(prog1)-1] = prog_re
    num_dead_nodes(nn2)
    if prog2[-1] > prog3[-1]:
        nn1.save()
        print("The original was better")
    else:
        nn2.save()
        print("The modified was better")
    plt.plot(np.arange(len(prog2)), prog2)
    plt.plot(np.arange(len(prog3)), prog3)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'data':
        res = forward_backward_analysis()
        print("Correct: %.3f%%"%(100*res[2]))
        plot_analysis(*res[:2])
        plot_images(res[3], np.arange(10))
    elif len(sys.argv) == 2 and sys.argv[1] == 'reac':
        reactivate_dead_nodes(Network())
    elif len(sys.argv) == 2 and sys.argv[1] == 'comp':
        print("Comparing a normal epoch and Reactivating nodes")
        compare_learning()
    elif len(sys.argv) == 3 and sys.argv[1] == 'comp' and sys.argv[2].isnumeric():
        print("Comparing a normal epoch and Reactivating nodes")
        compare_learning(int(sys.argv[2]))
    else:
        res = backward_analysis()
        plot_analysis(*res[:2])
        plot_images(res[3], [str(i) for i in range(10)]+['All'])
