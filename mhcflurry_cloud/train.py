from __future__ import absolute_import

import collections
import logging
import time
import itertools

import numpy
import pandas

import mhcflurry

from joblib import Parallel, delayed

from .scoring import make_scores

DEFAULT_MODEL = dict(
    max_ic50=50000,
    n_training_epochs=250,
    batch_size=128,
    impute=False,
    pretrain_decay="1 / (1+epoch)**2",
    fraction_negative=0.2,
    dropout_probability=0.5,
    embedding_output_dim=32,
    layer_sizes=[64],
    activation="tanh")

def models_grid(**kwargs):
    '''
    Make a "grid" of models by taking the cartesian product of all specified
    model parameter lists.

    Parameters
    -----------
    
    The valid paramters are the entries of DEFAULT_MODEL in this module.

    Each parameter must be a list giving the values to search across.

    Returns
    -----------
    list of dict giving the parameters for each model
    '''

    # Check parameters
    for (key, value) in kwargs.items():
        if key not in DEFAULT_MODEL:
            raise ValueError(
                "No such model parameters: %s. Valid parameters are: %s"
                % (key, " ".join(DEFAULT_MODEL)))
        if not isinstance(value, list):
            raise ValueError("All parameters must be lists, but %s is %s"
                % (key, str(type(value))))

    # Make models, using defaults from DEFAULT_MODEL.
    parameters = dict((key, [value]) for (key, value) in DEFAULT_MODEL.items())
    parameters.update(kwargs)
    parameter_names = list(parameters)
    parameter_values = [parameters[name] for name in parameter_names]

    models = [
        dict(zip(parameter_names, model_values))
        for model_values in itertools.product(*parameter_values)
    ]
    return models

def impute_and_select_allele(dataset, imputer, allele=None, **kwargs):
    '''
    Run imputation and optionally filter to the specified allele. 
    '''
    result = dataset.impute_missing_values(imputer, **kwargs)

    if allele is not None:
        result = result.get_allele(allele)
    return result

def train_and_test_one_model_one_fold(
        model_description,
        train_dataset,
        test_dataset=None,
        imputed_train_dataset=None,
        return_train_scores=True,
        return_predictor=False,
        return_train_predictions=False,
        return_test_predictions=False,
        n_jobs=1):
    '''
    Task for instantiating, training, and testing one model on one fold.

    Parameters
    -----------
    model_description : dict of model parameters

    train_dataset : mhcflurry.Dataset
        Dataset to train on. Must include only one allele.

    test_dataset : mhcflurry.Dataset, optional
        Dataset to test on. Must include only one allele. If not specified
        no testing is performed.

    imputed_train_dataset : mhcflurry.Dataset, optional
        Required only if model_description["impute"] == True

    return_train_scores : boolean
        Calculate and include in the result dict the auc/f1/tau scores on the
        training data.

    return_predictor : boolean
        Calculate and include in the result dict the trained predictor.

    return_train_predictions : boolean
        Calculate and include in the result dict the model predictions on the
        train data.

    return_test_predictions : boolean
        Calculate and include in the result dict the model predictions on the
        test data.

    Returns
    -----------
    dict
    '''
    assert len(train_dataset.unique_alleles()) == 1, "Multiple train alleles"
    allele = train_dataset.alleles[0]
    if test_dataset is not None:
        assert len(train_dataset.unique_alleles()) == 1, \
            "Multiple test alleles"
        assert train_dataset.alleles[0] == allele, \
            "Wrong test allele %s != %s" % (train_dataset.alleles[0], allele)
    if imputed_train_dataset is not None:
        assert len(imputed_train_dataset.unique_alleles()) == 1, \
            "Multiple imputed train alleles"
        assert imputed_train_dataset.alleles[0] == allele, \
            "Wrong imputed train allele %s != %s" % (
                imputed_train_dataset.alleles[0], allele)

    if model_description["impute"]:
        assert imputed_train_dataset is not None

    # Make a predictor
    model_params = dict(model_description)
    fraction_negative = model_params.pop("fraction_negative")
    impute = model_params.pop("impute")
    n_training_epochs = model_params.pop("n_training_epochs")
    pretrain_decay = model_params.pop("pretrain_decay")
    batch_size = model_params.pop("batch_size")
    max_ic50 = model_params.pop("max_ic50")

    logging.info(
        "%10s train_size=%d test_size=%d impute=%s model=%s" %
        (allele,
            len(train_dataset),
            len(test_dataset) if test_dataset is not None else 0,
            impute,
            model_description))

    predictor = mhcflurry.Class1BindingPredictor.from_hyperparameters(
        max_ic50=max_ic50,
        **model_params)

    # Train predictor
    fit_time = -time.time()
    predictor.fit_dataset(
        train_dataset,
        pretrain_decay=lambda epoch: eval(pretrain_decay, {
            'epoch': epoch, 'numpy': numpy}),
        pretraining_dataset=imputed_train_dataset if impute else None,
        verbose=False,
        batch_size=batch_size,
        n_training_epochs=n_training_epochs,
        n_random_negative_samples=int(fraction_negative * len(train_dataset)))
    fit_time += time.time()

    result = {
        'fit_time': fit_time,
    }    

    if return_predictor:
        result['predictor'] = predictor

    if return_train_scores or return_train_predictions:
        train_predictions = predictor.predict(train_dataset.peptides)
        if return_train_scores:
            result['train_scores'] = make_scores(
                train_dataset.affinities,
                train_predictions,
                max_ic50=model_description["max_ic50"])
        if return_train_predictions:
            result['train_predictions'] = train_predictions

    if test_dataset is not None:
        test_predictions = predictor.predict(test_dataset.peptides)
        result['test_scores'] = make_scores(
            test_dataset.affinities,
            test_predictions,
            max_ic50=model_description["max_ic50"])
        if return_test_predictions:
            result['test_predictions'] = test_predictions
    return result

def train_across_models_and_folds(
        folds,
        model_descriptions,
        cartesian_product_of_folds_and_models=True,
        return_predictors=False,
        n_jobs=1,
        verbose=0,
        pre_dispatch='2*n_jobs'):
    '''
    Train and optionally test any number of models across any number of folds.

    Parameters
    -----------
    folds : list of AlleleSpecificTrainTestFold

    model_descriptions : list of dict
        Models to test

    cartesian_product_of_folds_and_models : boolean, optional
        If true, then a predictor is treained for each fold and model
        description.
        If false, then len(folds) must equal len(model_descriptions), and
        the i'th model is trained on the i'th fold.

    return_predictors : boolean, optional
        Include the trained predictors in the result.

    n_jobs : integer, optional
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    verbose : integer, optional
        The joblib verbosity. If non zero, progress messages are printed. Above
        50, the output is sent to stdout. The frequency of the messages
        increases with the verbosity level. If it more than 10, all iterations
        are reported.

    pre_dispatch : {"all", integer, or expression, as in "3*n_jobs"}
        The number of joblib batches (of tasks) to be pre-dispatched. Default
        is "2*n_jobs". 

    Returns
    -----------
    pandas.DataFrame
    '''

    if cartesian_product_of_folds_and_models:
        model_and_fold_indices = [
            (fold_num, model_num)
            for fold_num in range(len(folds))
            for model_num in range(len(model_descriptions))
        ]
    else:
        assert len(folds) == len(model_descriptions), \
            "folds and models have different lengths and " \
            "cartesian_product_of_folds_and_models is False"

        model_and_fold_indices = [
            (num, num)
            for num in range(len(folds))
        ]

    logging.info("Training %d architectures on %d folds = %d predictors." % (
        len(model_descriptions), len(folds), len(model_and_fold_indices)))

    task_results = Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        pre_dispatch=pre_dispatch)(
            delayed(train_and_test_one_model_one_fold)(
                model_descriptions[model_num],
                train_dataset=folds[fold_num].train,
                test_dataset=folds[fold_num].test,
                imputed_train_dataset=folds[fold_num].imputed_train,
                return_predictor=return_predictors)
            for (fold_num, model_num) in model_and_fold_indices)

    logging.info("Done.")

    results_dict = collections.OrderedDict()

    def column(key, value):
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(value)

    for ((fold_num, model_num), task_result) in zip(
            model_and_fold_indices, task_results):

        fold = folds[fold_num]
        model_description = model_descriptions[model_num]

        column("allele", fold.allele)
        column("fold_num", fold_num)
        column("model_num", model_num)

        column("train_size", len(fold.train))

        if fold.test is not None:
            column("test_size", len(fold.test))

        if fold.imputed_train is not None:
            column("imputed_train_size", len(fold.imputed_train))

        # Scores
        for score_kind in ['train', 'test']:
            field = "%s_scores" % score_kind
            for (score, value) in task_result.pop(field, {}).items():
                column("%s_%s" % (score_kind, score), value)

        # Misc. fields
        for (key, value) in task_result.items():
            column(key, value)

        # Model parameters
        for (model_param, value) in model_description.items():
            column("model_%s" % model_param, value)

    results_df = pandas.DataFrame(results_dict)
    return results_df
