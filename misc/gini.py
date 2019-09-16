import numpy as np
from sklearn.preprocessing import binarize


def gini(fact, pred, classes):
    assert len(fact) == len(pred)
    len_ = len(fact)

    g_sum = 0.0
    for c in classes:
        pred_ = pred[fact == c]
        g_sum += len(fact[fact == c]) / len_ * len(pred_[pred[fact == c] != c]) / len_

    return g_sum


# print(gini(np.array([0, 0, 0, 0, 1]), np.array([0, 0, 0, 1, 0]), [0, 1]))
# exit(0)

fact = np.array([0, 0, 1, 0, 1]).reshape(-1, 1)
pred_prob = np.array([.2, .7, .6, .2, .75]).reshape(-1, 1)

# data = np.concatenate([fact, pred_prob, pred], axis=1)
# data = np.concatenate([np.array([3,3,3],dtype='int').reshape(1, -1), data], axis=0)

# threshold = np.ones((len(pred_prob) + 2, 1))
# threshold[0,0] = 0
# threshold[1:len(pred_prob)+1,0] = pred_prob

# Sorted ndarray of probabilities, with extra 0.0 and 1.0 at the ends.
tsh = np.vstack(([[0.]], np.sort(pred_prob, axis=0), [[1.]]))

# tuples of (threshold, gini impurity value)
tsh_gini = []
for i in range(1, tsh.shape[0]):
    threshold = np.mean((tsh[i - 1, 0], tsh[i, 0]))
    pred = binarize(pred_prob, threshold=threshold, copy=True).astype('int')
    gini_value = gini(fact.reshape(-1), pred.reshape(-1), [0, 1])
    tsh_gini.append((threshold, gini_value))

print(tsh_gini)

tsh_gini = np.array(tsh_gini)
best_threshold = tsh_gini[tsh_gini[:, 1] == np.amin(tsh_gini, axis=0)[1]][0][0]
print(best_threshold)