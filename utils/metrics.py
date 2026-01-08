import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def compute_metrics(y_true, y_pred):
    """Compute overall accuracy, precision, recall, F1 (macro)."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return acc, prec, rec, f1

def per_class_metrics(y_true, y_pred, labels=None):
    """Compute precision, recall, F1 for each class."""
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
    return prec, rec, f1, support

def compute_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
    """Compute confusion matrix (optionally normalized)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm

def classification_report_str(y_true, y_pred, target_names=None):
    """Return a classification report as a string."""
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
