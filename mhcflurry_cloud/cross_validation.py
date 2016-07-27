from __future__ import absolute_import

import collections
import logging

from joblib import Parallel, delayed

from mhcflurry.dataset import Dataset
import pepdata

from .common import AlleleSpecificTrainTestFold
from .train import impute_and_select_allele

gbmr4_transformer = pepdata.reduced_alphabet.make_alphabet_transformer("gbmr4")
def default_projector(peptide):
    """
    Given a peptide, return a list of projections for it. The projections are:
        - the gbmr4 reduced representation
        - for all positions in the peptide, the peptide with a "." replacing
          the residue at that position

    Peptides with overlapping projections are considered similar when doing
    cross validation.

    Parameters
    ----------
    peptide : string

    Returns
    ----------
    string list
    """
    def projections(peptide, edit_distance=1):
        if edit_distance == 0:
            return set([peptide])
        return set.union(*[
                projections(p, edit_distance - 1)
                for p in (
                    peptide[0:i] + "." + peptide[(i + 1):]
                    for i in range(len(peptide)))
        ])
    return sorted(projections(peptide)) + [gbmr4_transformer(peptide)]

def similar_peptides(set1, set2, projector=default_projector):
    """
    Given two sets of peptides, return a list of the peptides whose reduced
    representations are found in both sets.
    
    Parameters
    ----------
    projector : (string -> string) or (string -> string list)
        Function giving projection(s) of a peptide
        
    Returns
    ----------
    string list
        
    """
    result = collections.defaultdict(lambda: ([], []))
    for (index, peptides) in enumerate([set1, set2]):
        for peptide in peptides:
            projections = projector(peptide)
            if not isinstance(projections, list):
                projections = [projections]
            for projection in projections:
                result[projection][index].append(peptide)
    
    common = set()
    for (peptides1, peptides2) in result.values():
        if peptides1 and peptides2:
            common.update(peptides1 + peptides2)

    return sorted(common)

def cross_validation_folds(
        train_data,
        alleles=None,
        n_folds=3,
        drop_similar_peptides=False,
        imputer=None,
        impute_kwargs={
            'min_observations_per_peptide': 2,
            'min_observations_per_allele': 2,
        },
        n_jobs=1,
        verbose=0,
        pre_dispatch='2*n_jobs'):
    '''
    Split a Dataset into n_folds cross validation folds for each allele,
    optionally performing imputation.

    Parameters
    -----------
    train_data : mhcflurry.Dataset

    alleles : string list, optional
        Alleles to run cross validation on. Default: all alleles in
        train_data.

    n_folds : int, optional
        Number of cross validation folds for each allele.

    drop_similar_peptides : boolean, optional
        For each fold, remove peptides from the test data that are similar
        to peptides in the train data. Similarity is defined as in the
        similar_peptides function.

    imputer : fancyimpute.Solver, optional
        Imputer to use. If not specified, no imputation is done.

    impute_kwargs : dict, optional
        Additional kwargs to pass to mhcflurry.Dataset.impute_missing_values.

    Returns
    -----------
    list of AlleleSpecificTrainTestFold of length num alleles * n_folds

    '''
    if alleles is None:
        alleles = train_data.unique_alleles()

    result = []
    imputation_tasks = []
    for allele in alleles:
        logging.info("Allele: %s" % allele)
        cv_iter = train_data.cross_validation_iterator(
            allele, n_folds=n_folds, shuffle=True)
        for (all_allele_train_split, full_test_split) in cv_iter:
            peptides_to_remove = []
            if drop_similar_peptides:
                peptides_to_remove = similar_peptides(
                    all_allele_train_split.get_allele(allele).peptides,
                    full_test_split.get_allele(allele).peptides
                )

            if peptides_to_remove:
                test_split = full_test_split.drop_allele_peptide_lists(
                    [allele] * len(peptides_to_remove),
                    peptides_to_remove)
                logging.info(
                    "After dropping similar peptides, test size %d->%d" % (
                        len(full_test_split), len(test_split)))
            else:
                test_split = full_test_split
 
            if imputer is not None:
                imputation_tasks.append(delayed(impute_and_select_allele)(
                    all_allele_train_split.to_dataframe(),
                    imputer=imputer,
                    allele=allele,
                    **impute_kwargs))

            train_split = all_allele_train_split.get_allele(allele)
            fold = AlleleSpecificTrainTestFold(
                allele=allele,
                train=train_split,
                imputed_train=None,
                test=test_split)
            result.append(fold)

    if imputer is not None:
        #import pdb ; pdb.set_trace()
        imputation_results = Parallel(
            backend='threading',
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch)(imputation_tasks)

        result = [
            result_fold._replace(
                imputed_train=Dataset(imputation_result))
            for (imputation_result, result_fold)
            in zip(imputation_results, result)
        ]
    return result

