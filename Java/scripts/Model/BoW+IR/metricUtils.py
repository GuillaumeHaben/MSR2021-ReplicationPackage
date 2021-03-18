from sklearn.metrics import confusion_matrix
import numpy as np
# Scoring definitions

# TN
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

# FP
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

# FN
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

#TP
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

# Recall
def recall(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

# FPR
def fpr(y_true, y_pred): return fp(y_true, y_pred) / (fp(y_true, y_pred) + tn(y_true, y_pred))

# TPR
def tpr(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

# TNR
def tnr(y_true, y_pred): return 1 - fpr(y_true, y_pred)

# Precision
def precision(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fp(y_true, y_pred))

# F1
def f1(y_true, y_pred): return 2*tp(y_true, y_pred) / (2*tp(y_true, y_pred) + fp(y_true, y_pred) + fn(y_true, y_pred))

# AUC
def auc(y_true, y_pred): return (tpr(y_true, y_pred) + tnr(y_true, y_pred)) / 2

# MCC
def mcc(y_true, y_pred): return (tp(y_true, y_pred) * tn(y_true, y_pred) - fp(y_true, y_pred) * fn(y_true, y_pred)) / np.sqrt((tp(y_true, y_pred) + fp(y_true, y_pred)) * (tp(y_true, y_pred) + fn(y_true, y_pred)) * (tn(y_true, y_pred) + fp(y_true, y_pred)) * (tn(y_true, y_pred) + fn(y_true, y_pred)))
