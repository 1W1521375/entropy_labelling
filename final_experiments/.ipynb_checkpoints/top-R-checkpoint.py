import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import itertools
from random import shuffle
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
    else:
        print("sth wrong")
        
# top-k labelling
def topk_label(probas, s_cls, k):
    l_indexes = probas.argsort()[::-1][:k]
    labels = np.array([s_cls[i] for i in l_indexes])
    return labels

# labelling and evaluating them
def topk_scls_eval(part, classes, orig_A, lim_A):
    s_cls = classes

    # extract dataset of chosen classes
    trn_imgs = [img for i, img in enumerate(train_imgs) if train_labels[i] in s_cls]
    trn_labels = [label for label in train_labels if label in s_cls]
    tst_imgs = [img for i, img in enumerate(test_imgs) if test_labels[i] in s_cls]
    tst_labels = [label for label in test_labels if label in s_cls]

    # generate an annotator
    ann_model = LR().fit(trn_imgs[:orig_A], trn_labels[:orig_A])
    
    # for top-1.0
    if (part == 0):
        mul_labels = ann_model.predict(trn_imgs[orig_A:orig_A + lim_A])
        # labels score evaluation
        score = 0
        for label, t_label in zip(mul_labels, trn_labels[orig_A:orig_A + lim_A]):
            if (label == t_label):
                score += 1
        
        return (1.0, score*100/len(mul_labels), score*100/lim_A)
    
    # for top-R (1.05, 1.1, ..., 1.25)
    else:
        # split data to label into two groups
        f_imgs, s_imgs, f_labels, s_labels = train_test_split(trn_imgs[orig_A:orig_A + lim_A], trn_labels[orig_A:orig_A + lim_A], test_size = part/100)
        # top-1
        f_ann_proba = ann_model.predict_proba(f_imgs)
        ord_labels = [topk_label(probas, s_cls, 1) for probas in f_ann_proba]
        # top-2
        s_ann_proba = ann_model.predict_proba(s_imgs)
        top2_labels = [topk_label(probas, s_cls, 2) for probas in s_ann_proba]
        # concat top-1 results and top-2 results
        mul_labels = ord_labels + top2_labels
        
        # labels score evaluation
        score = 0
        for labels, t_label in zip(mul_labels, f_labels + s_labels):
            for l in labels:
                if (l == t_label):
                    score += 1

        m_labels = []
        for labels in mul_labels:
             [m_labels.append(l) for l in labels]                
                    
        return (len(m_labels)/lim_A, score*100/len(m_labels), score*100/lim_A)

# loading MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imgs = np.array([x.ravel() for x in train_images])
test_imgs = np.array([y.ravel() for y in test_images])
img_SIZE = train_images.shape[1]*train_images.shape[2]

# main experiment
classes = [i for i in range(10)]
orig_A1, lim_A1 = 2000, 2000
fact_10 = factorial(10)

# main experiment
classes = [i for i in range(10)]
orig_A1, lim_A1 = 2000, 2000

fact_10 = factorial(10)

lqual_mnist = []

Rs = [i/100 for i in range(100, 130, 5)]
for R in Rs:
    mnist_evals = []
    part = round((R - 1.00)*100)
    for i in range (2, 11): # i: num of sub-classes
        print(f"{i} classes, top-{R}, {part}% of data mul-labelled")
        combi_ni = fact_10//(factorial(i)*factorial(10 - i))
        a, b, c = 0, 0, 0
        if (i == 10):
            d, e, f = 0, 0, 0
            for _ in range(5):
                sample_lnum, sample_lqual, sample_lqual2 = topk_scls_eval(part, list(scls), orig_A1, lim_A1)
                mnist_evals.append((sample_lnum, sample_lqual, sample_lqual2))
        else:
            for scls in itertools.combinations(classes, i):
                x, y, z = 0, 0, 0
                for _ in range(5):
                    s, t, u = topk_scls_eval(part, list(scls), orig_A1, lim_A1)
                    x += s
                    y += t
                    z += u
                sample_lnum, sample_lqual, sample_lqual2 = x/5, y/5, z/5
                a += sample_lnum
                b += sample_lqual
                c += sample_lqual2
            mnist_evals.append((a/combi_ni, b/combi_ni, c/combi_ni))
    print(f"{q}\n{mnist_evals}", sep = '\n', file = codecs.open("/home/k.goto/entropy_labelling/final_experiments/top-R_results.txt", 'a', 'utf-8'))
    quals = [e[1] for e in mnist_evals]
    lqual_mnist.append(quals)