from sklearn.metrics import accuracy_score
import numpy as np 

def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    preds = np.argmax(predictions, axis=1).reshape(-1)
    labels = np.zeros(preds.shape)
    value_mean = np.average(predictions, axis=0)
    metric = {
                "accuracy": float(
                    accuracy_score(labels, preds, normalize=True)
                ),
                "accepts_end_token_value_mean": value_mean[0],
                "rejects_end_token_value_mean": value_mean[1]
            }
    return metric




