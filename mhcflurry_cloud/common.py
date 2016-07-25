from __future__ import absolute_import

import collections

AlleleSpecificTrainTestFold = collections.namedtuple(
    "AlleleSpecificTrainTestFold",
    "allele train imputed_train test")
