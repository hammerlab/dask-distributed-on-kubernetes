import mhcflurry
import fancyimpute

from . import data_path

from mhcflurry_cloud import model_selection

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
