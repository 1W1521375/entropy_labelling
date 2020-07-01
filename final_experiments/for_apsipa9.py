# for subclasses, over 80% ann
# Top-K (q = -inf, +inf included)

import matplotlib as mpl
import matplotlib.pyplot as plt
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
        
# top-k labelling
def topk_label(probas, s_cls, k):
    l_indexes = probas.argsort()[::-1][:k]
    labels = [s_cls[i] for i in l_indexes]
    return labels
    
# labelling and evaluating them
def info_trans_scoring_1(k, classes, orig_A, lim_A):
    s_cls = classes

    # extract dataset of chosen classes
    trn_imgs = [img for i, img in enumerate(train_imgs) if train_labels[i] in s_cls]
    trn_labels = [label for label in train_labels if label in s_cls]

    # generate an annotator
    a1_model = LR().fit(trn_imgs[:orig_A], trn_labels[:orig_A])
    a1_probas = a1_model.predict_proba(trn_imgs[orig_A:orig_A + lim_A])

    # top-k labelling
    mul_labels = [topk_label(probas, s_cls, k) for probas in a1_probas]

    # scoring info transmission
    score = 0
    for labels, probas in zip(mul_labels, a1_probas):
        u_dist = 1/len(labels)
        for i, label in enumerate(labels):
            score += np.power(probas[i] - u_dist, 2)
    
    return score

# loading MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imgs = np.array([x.ravel() for x in train_images])
test_imgs = np.array([y.ravel() for y in test_images])
img_SIZE = train_images.shape[1]*train_images.shape[2]

# main measurement
classes = [i for i in range(10)]
orig_A1, lim_A1 = 2000, 2000

fact_10 = factorial(10)

# from top-1 (ord) to top-8
for k in range(1, 9):
    results = []
    i = k + 1
    while (i < 10): # i: num of sub-classes
        temp = 0
        combi_ni = fact_10//(factorial(i)*factorial(10 - i))
        for scls in itertools.combinations(classes, i):
            temp += info_trans_scoring_1(k, list(scls), orig_A1, lim_A1)
        results.append(temp/combi_ni)
        i += 1
    print(f"Top-{k}\n{results}", sep = "\n", file = codecs.open("for_apsipa6.txt", 'a', 'utf-8'))
