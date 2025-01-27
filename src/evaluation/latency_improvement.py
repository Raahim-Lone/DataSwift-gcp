import numpy as np
import logging

logger = logging.getLogger(__name__)

def calcLatencyImprovement(W,H,X,Y,P,M):
    """
    Given W,H,X,Y,P,M compute latency improvement:
    For each query q:
      baseline = average actual latency of observed hints
      choose hint with minimal predicted latency
      improvement = (baseline - chosen_latency)/baseline
    """
    XW = X @ W
    YH = Y @ H
    num_queries = P.shape[0]
    improvements = []

    for q in range(num_queries):
        observed_indices = np.where(M[q,:]>0)[0]
        if len(observed_indices)==0:
            continue
        baseline = np.mean(P[q, observed_indices])
        preds = np.array([(XW[q,:] * YH[j,:]).sum() for j in observed_indices])
        best_idx = observed_indices[np.argmin(preds)]
        chosen_latency = P[q,best_idx]
        if baseline>0:
            improvements.append((baseline - chosen_latency)/baseline)

    improvement = np.mean(improvements) if improvements else 0.0
    logger.debug(f"Calculated latency improvement: {improvement}")
    return improvement
