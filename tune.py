from __future__ import division
from sklearn.metrics import confusion_matrix
import pandas

def frange(start, stop, step):
    if start > stop:
        i = start
        while i > stop:
            yield i
            i -= step
    else:
        i = start
        while i < stop:
            yield i
            i += step

def calculate_cm_stats(binary_y, df, x):
    cm = confusion_matrix(binary_y, (df > x)).ravel()
    _tn, _fp, _fn, _tp = cm
    _totals = _tn + _fp + _fn + _tp
    _accuracy = float(_tp+_tn)/_totals
    _fn_inverse = _fn/_totals
    _percent_fn = (100-(100*_fn_inverse))/100
    _fn_to_accuracy = (_accuracy + _percent_fn)/2
    _weighted_fn_to_accuracy = float(((_percent_fn * 2) + _accuracy)/3)
    _percent_detected = (_tp)/(_tp+_fn)
    _weighted_detection_to_accuracy = ((_percent_detected + _accuracy) / 2)
    _lambda = (_weighted_detection_to_accuracy + _weighted_fn_to_accuracy) / 2
    return {
                "tn":_tn,"fp":_fp,"fn":_fn,"tp":_tp,
                "total":_totals,
                "accuracy":_accuracy,
                "fn_inverse":_fn_inverse,
                "percent_fn":_percent_fn,
                "fn_to_accuracy": _fn_to_accuracy,
                "weighted_fn_to_accuracy":_weighted_fn_to_accuracy,
                "percent_detected":_percent_detected,
                "weighted_detection_to_accuracy":_weighted_detection_to_accuracy,
                "lambda":_lambda,
            }

def gd_threshold(df, binary_y, best = 0, variable = "accuracy", verbose = False):
    best_x = 0
    for x in frange(1.0, 0, 0.01):
        stats = calculate_cm_stats(binary_y, df, x)
        if stats["weighted_fn_to_accuracy"] > best and variable == "fn_weighted":
            best = stats["weighted_fn_to_accuracy"]
            best_x = x
        if stats["fn_to_accuracy"] > best and variable == "fn_to_accuracy":
            best = stats["fn_to_accuracy"]
            best_x = x
        if stats["accuracy"] > best and variable == "accuracy":
            best = stats["accuracy"]
            best_x = x
        if  stats["weighted_detection_to_accuracy"] > best and variable == "detection":
            best = stats["weighted_detection_to_accuracy"]
            best_x = x
        if  stats["lambda"] > best and variable == "lambda":
            best = stats["lambda"]
            best_x = x

    if verbose:
        stats = calculate_cm_stats(binary_y, df, best_x)
        print("Best Threshold for {}: {}".format(variable, best_x))
        print confusion_matrix(binary_y, (df > best_x)).ravel()
        print(stats)
    return best_x, best
