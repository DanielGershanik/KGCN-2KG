from numpy.lib.function_base import append
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


class Metrics():
  def __init__(self):
      self.tp, self.fn, self.fp, self.tn = 0, 0, 0, 0
      self.tpr, self.tnr = [], []
      self.acc = 0

  def confusion_metrix(self, labels, predictions):
    result = confusion_matrix(labels, predictions)

    tp, fn, fp, tn = result[0][0], result[0][1], result[1][0], result[1][1]

    self.tp += tp
    self.fn += fn
    self.fp += fp
    self.tn += tn
    
  def roc_accuracy(self, labels, predictions):
    result = roc_auc_score(labels, predictions)
    self.acc += result

  def topn_acc(self, lables, predictions, n=5):
    topn = np.argsort(predictions, axis = 0)[-n:]
    topn_val = np.mean(np.array([1 if lables[k] in topn[k] else 0 for k in range(len(topn))]))

    pass
    # return np.mean(np.array([1 if lables[k] in topn[k] else 0 for k in range(len(topn))]))

  def get_acc(self):
    return self.acc

  def get_conf(self):
    self.tpr, self.tnr = self.tp / (self.tp + self.fn), self.tn / (self.tn + self.fp)
    return self.tpr, self.tnr


# def top_n_accuracy(X,y,n,classifier):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     vectorizer = TfidfVectorizer(min_df=2)
#     X_train_sparse = vectorizer.fit_transform(X_train)
#     feature_names = vectorizer.get_feature_names()
#     test = vectorizer.transform(X_test)
#     clf = classifier
#     clf.fit(X_train_sparse,y_train)
#     predictions = clf.predict(test)
#     probs = clf.predict_proba(test)
#     topn = np.argsort(probs, axis = 1)[:,-n:]
#     y_true = np.array(y_test)
#     return np.mean(np.array([1 if y_true[k] in topn[k] else 0 for k in range(len(topn))]))