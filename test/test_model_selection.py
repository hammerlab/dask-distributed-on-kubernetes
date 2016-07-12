from nose.tools import eq_

import celery
from celery.signals import task_success

import mhcflurry
import fancyimpute

from . import data_path

from mhcflurry_cloud import model_selection

def init_celery():
    celery.finalize()
    if celery.conf.CELERY_ALWAYS_EAGER:
        task_success.connect(check_result_serialization)

def check_result_serialization(result, **kwargs):
    celery.backend.encode(dict(result=result))

def test_imputation():
    imputer = fancyimpute.MICE(
        n_imputations=2, n_burn_in=1, n_nearest_columns=25)
    train_data = mhcflurry.dataset.Dataset.from_csv(
        data_path("bdata.2009.mhci.public.1.txt"))

    result = model_selection.impute.delay(
        train_data,
        imputer,
        min_observations_per_peptide=20,
        min_observations_per_allele=20).get()
    assert result is not None

'''
def Xtest_imputation_serialization():
    class CeleryConfig(object):
        BROKER_URL = 'sqla+sqlite:////tmp/celerydb.sqlite'
        CELERY_RESULT_BACKEND = 'sqlite:////tmp/results.sqlite'

    model_selection.celery_app.config_from_object(CeleryConfig)
    try:
        test_imputation()
    finally:
        model_selection.celery_app.config_from_object(
            model_selection.LocalCeleryConfig)
'''

        