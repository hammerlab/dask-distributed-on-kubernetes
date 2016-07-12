import fancyimpute

import mhcflurry
import mhcflurry_cloud.model_selection

imputer = fancyimpute.MICE(
    n_imputations=2, n_burn_in=1, n_nearest_columns=25)

train_data = mhcflurry.dataset.Dataset.from_csv(
    "test/data/bdata.2009.mhci.public.1.txt")

result = mhcflurry_cloud.model_selection.impute.delay(
    train_data,
    imputer,
    min_observations_per_peptide=20,
    min_observations_per_allele=20)

print(result)
print(result.get())