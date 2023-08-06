from sklearn.metrics import accuracy_score
import numpy as np 

def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

def compute_metrics(eval_preds):
    preds, _ = eval_preds
    preds = np.argmax(preds, axis=0).reshape(-1)
    labels = np.zeros(preds.shape)
    return accuracy(predictions=preds, references=labels)




