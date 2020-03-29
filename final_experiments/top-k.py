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
    else:
        print("sth wrong")
        
# top-k labelling
def topk_label(probas, s_cls, k):
    l_indexes = probas.argsort()[::-1][:k]
    labels = np.array([s_cls[i] for i in l_indexes])
    return labels

# labelling and evaluating them
def topk_scls_eval(k, classes, orig_A, lim_A):
    s_cls = classes

    # extract dataset of chosen classes
    trn_imgs = [img for i, img in enumerate(train_imgs) if train_labels[i] in s_cls]
    trn_labels = [label for label in train_labels if label in s_cls]
    tst_imgs = [img for i, img in enumerate(test_imgs) if test_labels[i] in s_cls]
    tst_labels = [label for label in test_labels if label in s_cls]

    # generate an annotator
    a1_model = LR().fit(trn_imgs[:orig_A], trn_labels[:orig_A])
    a1_proba = a1_model.predict_proba(trn_imgs[orig_A:orig_A + lim_A])

    # entropy labelling
    mul_labels = [topk_label(probas, s_cls, k) for probas in a1_proba]
    
    # labels score evaluation
    score = 0
    for labels, t_label in zip(mul_labels, trn_labels[orig_A:orig_A + lim_A]):
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

lqual_mnist = []

# from top-1 (ord) to top-9
for k in range(1, 10):
    mnist_evals = []
    i = k + 1
    while (i < 11): # i: num of sub-classes
        print(f"{i} classes, {k}-labels")
        a, b, c = 0, 0, 0
        if (i == 10):
            sample_lnum, sample_lqual, sample_lqual2 = topk_scls_eval(k, classes, orig_A1, lim_A1)
            mnist_evals.append((sample_lnum, sample_lqual, sample_lqual2))
        else:
            combi_ni = fact_10//(factorial(i)*factorial(10 - i))
            for scls in itertools.combinations(classes, i):
                sample_lnum, sample_lqual, sample_lqual2 = topk_scls_eval(k, list(scls), orig_A1, lim_A1)
                a += sample_lnum
                b += sample_lqual
                c += sample_lqual2
            mnist_evals.append((a/combi_ni, b/combi_ni, c/combi_ni))
        i += 1
    print(mnist_evals, sep = "\n", file = codecs.open("/home/killerqueen/lab/final_experiments/top-k_results.txt", "a", "utf-8"))
            
    quals = [e[1] for e in mnist_evals]
    lqual_mnist.append(quals)

plt.figure(dpi = 100)
plt.title("Quality of labels; Top-k (k = 1, 2, 3, 4, 5, 6, 7, 8, 9)")
plt.plot([i for i in range(2, 11)], lqual_mnist[0], marker = 'o', color = 'b', label = "Top-1")
plt.plot([i for i in range(3, 11)], lqual_mnist[1], marker = 'o', color = 'g', label = "Top-2")
plt.plot([i for i in range(4, 11)], lqual_mnist[2], marker = 'o', color = 'r', label = "Top-3")
plt.plot([i for i in range(5, 11)], lqual_mnist[3], marker = 'o', color = 'c', label = "Top-4")
plt.plot([i for i in range(6, 11)], lqual_mnist[4], marker = 'o', color = 'm', label = "Top-5")
plt.plot([i for i in range(7, 11)], lqual_mnist[5], marker = 'o', color = 'y', label = "Top-6")
plt.plot([i for i in range(8, 11)], lqual_mnist[6], marker = 'o', color = 'k', label = "Top-7")
plt.plot([i for i in range(9, 11)], lqual_mnist[7], marker = 'o', color = 'b', label = "Top-8")
plt.plot([i for i in range(10, 11)], lqual_mnist[8], marker = 'o', color = 'g', label = "Top-9")
plt.xlabel("Num of sub-classes")
plt.ylabel("Average label accuracy of labels generated")	
plt.legend(loc = 'best')
plt.grid(True)
plt.savefig("/home/killerqueen/lab/final_experiments/topk_labels-acc.pdf")
