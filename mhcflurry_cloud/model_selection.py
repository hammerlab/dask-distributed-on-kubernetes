from __future__ import absolute_import

import logging

from mhcflurry_cloud.celery import app

@app.task
def impute(dataset, imputer, allele=None, **kwargs):
    '''
    Celery wrapper for imputation.
    '''
    result = dataset.impute_missing_values(imputer, **kwargs)
    if allele is not None:
        result = result.get_allele(allele)
    return result

def make_cv_folds(
        train_data,
        alleles=None,
        n_folds=3,
        shuffle=True,
        drop_similar_peptides=False,
        impute=False,
        imputer_name="MICE",
        imputer_kwargs=dict(
            n_imputations=50, n_burn_in=5, n_nearest_columns=25)):
    '''
    Parameters
    -----------
        train_data : mhcflurry.Dataset


    Returns
    -----------
    dict mapping allele to list of (train set, imputed train set, test set) for
    each fold
    '''
    if alleles is None:
        alleles = train_data.alleles

    cv_splits = {}
    for allele in alleles:
        print("Allele: %s" % allele)
        cv_iter = train_data.cross_validation_iterator(
            allele, n_folds=n_folds, shuffle=shuffle)
        triples = []
        for (all_allele_train_split, full_test_split) in cv_iter:
            peptides_to_remove = similar_peptides(
                all_allele_train_split.get_allele(allele).peptides,
                full_test_split.get_allele(allele).peptides
            )
            logging.info("Peptides to remove: %d: %s" % (
                len(peptides_to_remove), str(peptides_to_remove)))
            if peptides_to_remove:
                test_split = full_test_split.drop_allele_peptide_lists(
                    [allele] * len(peptides_to_remove),
                    peptides_to_remove)
                logging.info(
                    "After dropping similar peptides, test size %d -> %d" % (
                        len(full_test_split), len(test_split)))
            else:
                test_split = full_test_split
            imputed_train_split = all_allele_train_split.impute_missing_values(
                imputer,
                min_observations_per_peptide=2,
                min_observations_per_allele=2).get_allele(allele)
            train_split = all_allele_train_split.get_allele(allele)
            triples.append((train_split, imputed_train_split, test_split))
        cv_splits[allele] = triples 
    return cv_splits