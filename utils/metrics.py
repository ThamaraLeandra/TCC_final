from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix
)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def compute_diagnostics(y_true, y_pred):
    prec_c, rec_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    return {
        "per_class_precision": prec_c.tolist(),
        "per_class_recall": rec_c.tolist(),
        "per_class_f1": f1_c.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
