import functools
import numpy as np
import scipy
import math
import sklearn
from logging import getLogger
from transformers import EvalPrediction
from typing import Callable, Dict, List, Tuple

logger = getLogger(__name__)




def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    pearson_corrcoef = scipy.stats.pearsonr(targets, predictions.squeeze())[0]

    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson_corrcoef": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    spearman_corrcoef = scipy.stats.spearmanr(targets, predictions.squeeze())[0]

    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearman_corrcoef": spearman_corrcoef}


def build_compute_metrics_fn(task_name: str, groups: List[str]) -> Callable[[EvalPrediction], Dict]:

    def compute_metrics(pred: EvalPrediction, metrics) -> Dict:
        pred_str, label_str = pred.predictions, pred.label_ids
        results = {}
        for metric in metrics:
            results.update(metric(pred_str, label_str))
        return results

    def tasks_metrics(task) -> Dict:
        from hyperpersonalisation.data import TASK_MAPPING
        return functools.partial(compute_metrics, metrics=TASK_MAPPING[task].metrics)

    return {group: tasks_metrics(task_name) for group in groups}
