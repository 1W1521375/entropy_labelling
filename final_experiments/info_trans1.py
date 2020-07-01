# for subclasses, over 80% ann
# q = -1.0, -0.1, 0, 0.1, 1

import matplotlib as mpl
import matplotlib.pyplot as plt
from math import isnan
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from keras.datasets import mnist
import itertools
import codecs
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action = 'ignore', category = FutureWarning)

def factorial(n):
    if n == 0:
        return 1
    elif n > 0:
        return n*factorial(n - 1)
        
# tsallis entropy labelling
def tsallis_label(q, probas, s_cls):
    elements = np.power(probas, q - 1)
    # thrshld in tsallis entropy model
    ts_thrshld = np.sum(np.power(probas, q))
    if q < 1:
        labels = [s_cls[i] for i, e in enumerate(elements) if e < ts_thrshld]
    else:
        labels = [s_cls[i] for i, e in enumerate(elements) if e > ts_thrshld]
    
    # for a definite instance
    if (labels == []):
        labels = [s_cls[np.argmax(probas)]]
    
    return labels

# shannon entropy labelling
def shannon_label(probas, s_cls):
    info_con = (-1)*np.log2(probas)
    # entropy
    Hp = np.sum(np.multiply(probas, info_con))
    if isnan(Hp):
        labels = [s_cls[np.argmax(probas)]]
    else:
        labels = [s_cls[i] for i, Ipk in enumerate(info_con) if Ipk <= Hp]
    return labels

# 1/M labelling
def oneMth_label(probas, s_cls):
    # 1/M
    thrshld = 1/len(s_cls)
    labels = [s_cls[i] for i, pk in enumerate(probas) if pk > thrshld]
    return labels

# labelling and evaluating them
def info_trans_scoring(q, classes, orig_A, lim_A):
    s_cls = classes

    # extract dataset of chosen classes
    trn_imgs = [img for i, img in enumerate(train_imgs) if train_labels[i] in s_cls]
    trn_labels = [label for label in train_labels if label in s_cls]

    # generate an annotator
    a1_model = LR().fit(trn_imgs[:orig_A], trn_labels[:orig_A])
    a1_probas = a1_model.predict_proba(trn_imgs[orig_A:orig_A + lim_A])

    # labelling
    if (q == 0): # 1/M
        mul_labels = [oneMth_label(probas, s_cls) for probas in a1_probas]
    else if (q == 1): # shannon
        mul_labels = [shannon_label(probas, s_cls) for probas in a1_probas]
    else:
        mul_labels = [tsallis_label(q, probas, s_cls) for probas in a1_probas]

    # scoring info transmission
    score = 0
    for labels, probas in zip(mul_labels, a1_probas):
        u_dist = 1/len(labels)
        for label in labels:
            score += np.power(probas[label] - u_dist, 2)
    
    return score

# loading MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imgs = np.array([x.ravel() for x in train_images])
test_imgs = np.array([y.ravel() for y in test_images])

# main measurement
classes = [i for i in range(10)]
orig_A1, lim_A1 = 2000, 2000
fact_10 = factorial(10)
q_list = [-1.0, -0.1, 0, 0.1, 1.0]

for i in range(2, 10):
    results = []
    for q in q_list:
        temp = 0
        combi_ni = fact_10//(factorial(i)*factorial(10 - i))
        for scls in itertools.combinations(classes, i):
            temp += info_trans_scoring(q, scls, orig_A1, lim_A1)
        results.append(temp/combi_ni)
    print(f"{i}-classes: {results}", sep = "\n", file = codecs.open("info_trans.txt", 'a', 'utf-8'))