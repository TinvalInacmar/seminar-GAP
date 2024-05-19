from .builder import DATASETS

from .base_condensed_dataset import BaseCondensedDataset

@DATASETS.register_module()
class FERPlusCondensed(BaseCondensedDataset):
    pass