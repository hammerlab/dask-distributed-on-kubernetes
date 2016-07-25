from __future__ import absolute_import

from nose.tools import eq_ 

import mhcflurry
import fancyimpute

from . import data_path

import mhcflurry_cloud

# Run all tasks locally.
mhcflurry_cloud.celery.app.conf.update(
    CELERY_ALWAYS_EAGER=True,
)

def test_cross_validation():
    imputer = fancyimpute.MICE(
        n_imputations=2, n_burn_in=1, n_nearest_columns=25)
    train_data = (
        mhcflurry.dataset.Dataset.from_csv(
            data_path("bdata.2009.mhci.public.1.txt"))
        .get_alleles(["HLA-A0201", "HLA-A0202", "HLA-A0301"]))

    folds = mhcflurry_cloud.cross_validation_folds(
        train_data,
        n_folds=3,
        imputer=imputer,
        drop_similar_peptides=True,
        alleles=["HLA-A0201", "HLA-A0202"],
    )

    eq_(set(x.allele for x in folds), {"HLA-A0201", "HLA-A0202"})
    eq_(len(folds), 6)

    for fold in folds:
        eq_(fold.train.unique_alleles(), set([fold.allele]))
        eq_(fold.imputed_train.unique_alleles(), set([fold.allele]))
        eq_(fold.test.unique_alleles(), set([fold.allele]))

    models = mhcflurry_cloud.models_grid(
        activation=["tanh", "relu"],
        layer_sizes=[[4]],
        embedding_output_dim=[8],
        n_training_epochs=[3])
    print(models)

    df = mhcflurry_cloud.train_and_test_across_models_and_folds(
        folds,
        models)
    print(df)
    assert df.test_auc.mean() > 0.6
