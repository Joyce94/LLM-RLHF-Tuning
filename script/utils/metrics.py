from sklearn.metrics import accuracy_score
import numpy as np 

def compute_metrics_for_pair(eval_preds):
    predictions = eval_preds.predictions
    preds = np.argmax(predictions, axis=1).reshape(-1)
    labels = np.zeros(preds.shape)
    metric = {
                "accuracy": float(
                    accuracy_score(labels, preds, normalize=True)
                ),
            }
    return metric


