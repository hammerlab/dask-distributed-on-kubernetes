from __future__ import absolute_import

import sklearn
import numpy
import scipy

import mhcflurry

def make_scores(
        ic50_y,
        ic50_y_pred,
        sample_weight=None,
        threshold_nm=500,
        max_ic50=50000):
    """
    Calculate auc, f1, tau scores.    
    """

    y_pred = mhcflurry.regression_target.ic50_to_regression_target(
        ic50_y_pred, max_ic50)
    try:
        auc = sklearn.metrics.roc_auc_score(
            ic50_y <= threshold_nm,
            y_pred,
            sample_weight=sample_weight)
    except ValueError:
        auc = numpy.nan
    try:
        f1 = sklearn.metrics.f1_score(
            ic50_y <= threshold_nm,
            ic50_y_pred <= threshold_nm,
            sample_weight=sample_weight)
    except ValueError:
        f1 = numpy.nan
    try:
        tau = scipy.stats.kendalltau(ic50_y_pred, ic50_y)[0]
    except ValueError:
        tau = numpy.nan
    
    return dict(
        auc=auc,
        f1=f1,
        tau=tau,
    )    
