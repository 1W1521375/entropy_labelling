import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from keras.datasets import mnist
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
        
# entropy labelling
def tsallis_label(q, probas, s_cls):
    elements = np.power(probas, q - 1)
    # thrshld in tsallis entropy model
    ts_thrshld = np.sum(np.power(probas, q))
    if q < 1:
        labels = [s_cls[i] for i, e in enumerate(elements) if e <= ts_thrshld]
    else:
        labels = [s_cls[i] for i, e in enumerate(elements) if e >= ts_thrshld]
    return labels

def tsallis_scls_eval(q, classes, orig_A, lim_A):
    s_cls = classes
    
    # confusing pairs ... see LR-misclassification-habits.pdf
    conf_pairs = conf_pairs = [(0, 6), (3, 5), (4, 9), (7, 9), (8, 2, 5, 9)]
    # decide what to flip
    flip_dict = dict()
    for c_pair in conf_pairs:
    	origin = c_pair[0]
    	# if origin label is in sub_classes
    	if (origin in s_cls):
            intersection = set(c_pair)&set(s_cls)
            n = len(intersection) - 1
            # if there is at least one labels pair to flip
            if (1 <= n):
            	dests = intersection - {origin}
            	flip_dict[str(origin)] = list(dests)
    
    # extract dataset of the chosen classes
    trn_imgs = [img for i, img in enumerate(train_imgs) if train_labels[i] in s_cls]
    trn_labels = [label for label in train_labels if label in s_cls] 
    if (len(flip_dict) != 0):
        # use #(orig_A) images for generating an annotator
        orig_labels = trn_labels[:orig_A]
        imgs, labels = [], []
        # flip labels
        for c in s_cls:
            tr_imgs, tr_labels = [], []
            for i, l in enumerate(orig_labels):
                if (l == c):
                    tr_imgs.append(trn_imgs[i])
                    tr_labels.append(l)
            # half is kept as original, the other half is uniformly flipped
            dests = flip_dict.get(str(c))
            if (dests == None):
                imgs = imgs + tr_imgs
                labels = labels + tr_labels
                continue

            n = len(dests)
            half = len(tr_labels)//2
            chunk = half//n
            flipped_list = tr_labels[:half]
            # flipping
            for i in range(n):
                flipped_list = flipped_list + [dests[i] for k in range(chunk)]
            # 要素数がflip候補数で割り切れない場合は詰め物する(最後の要素繰り返すだけ)
            d = len(tr_labels) - len(flipped_list)
            while (d > 0):
                flipped_list.append(dests[-1])
                d -= 1
            # shuffling makes it equivalent to random sampling for flippling
            imgs = imgs + tr_imgs
            shuffle(flipped_list) # 破壊的操作，shuffleされたlistそのものが返ってくるのではない!
            labels = labels + flipped_list
    else:
        imgs = trn_imgs[:orig_A]
        labels = trn_labels[:orig_A]

    # generate an annotator
    a1_model = LR().fit(imgs, labels)
    a1_proba = a1_model.predict_proba(trn_imgs[orig_A:orig_A + lim_A])

    # entropy labelling
    mul_labels = [tsallis_label(q, probas, s_cls) for probas in a1_proba]
    
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

# from q \in [0.1, 0.5, 2.0, 10.0] (0, 1.0, +inf are substitued by 1/K, Shannon, Ord)
q_list = [0.1, 0.5, 2.0, 10.0]
for q in q_list:
    mnist_evals = []
    for i in range(2, 11): # i: num of sub-classes
        a, b, c = 0, 0, 0
        if (i == 10):
            for _ in range(5):
                sample_lnum, sample_lqual, sample_lqual2 = tsallis_scls_eval(q, classes, orig_A1, lim_A1)
                d += sample_lnum
                e += sample_lqual
                f += sample_lqual2
            mnist_evals.append((d/5, e/5, f/5))
        else:
            combi_ni = fact_10//(factorial(i)*factorial(10 - i))
            for scls in itertools.combinations(classes, i):
                a_pre, b_pre, c_pre = 0, 0, 0
                for _ in range(5):
                    sample_lnum, sample_lqual, sample_lqual2 = tsallis_scls_eval(q, list(scls), orig_A1, lim_A1)
                    a_pre += sample_lnum
                    b_pre += sample_lqual
                    c_pre += sample_lqual2
                a += a_pre/5
                b += b_pre/5
                c += c_pre/5
            mnist_evals.append((a/combi_ni, b/combi_ni, c/combi_ni))
    print(f"{q}\n{mnist_evals}", sep = '\n', file = codecs.open("/home/k.goto/entropy_labelling/results/txt_format/tsallis_flip.txt", 'a', 'utf-8'))

#     quals = [e[1] for e in mnist_evals]
#     lqual_mnist.append(quals)
    
# draw a graph
# plt.figure(dpi = 100)
# plt.title("Quality of labels; Tsallis (q = 0.1, ..., 2.0)")
# classes = [i for i in range(2, 11)]

# for c in range(20):
#     q_val = "q = " + str(q_list[c])
#     plt.plot(classes, lqual_mnist[c], marker = 'o', color = cm.bone(c/20), label = q_val)

# plt.xlabel("Num of sub-classes")
# plt.ylabel("Average label accuracy of labels generated")	
# plt.legend(loc = 'best')
# plt.grid(True)
# plt.savefig("/home/killerqueen/lab/final_experiments/tsallis_labels-acc_full.pdf")
