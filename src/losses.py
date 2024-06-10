import numpy as np
from sklearn.metrics import mean_squared_error

def MCRMSE(y_trues, y_preds):
    y_trues = np.asarray(y_trues)
    y_preds = np.asarray(y_preds)

    scores = []
    idxs = y_trues.shape[1]

    for i in range(idxs):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]

        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores.append(rmse)
    mcmrse_score = np.mean(scores)
    return mcmrse_score
