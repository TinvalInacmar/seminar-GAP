from .builder import DATASETS

from .base_fer_dataset import BaseFerDataset
from .base_condensed_dataset import BaseCondensedDataset
from .base_negative_dataset import BaseNegativeDataset


@DATASETS.register_module()
class FERPlus(BaseFerDataset):
    DATASET_CLASSES = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    CONTEMPT = True
    CONVERT_TABLE = (6, 4, 5, 3, 0, 1, 2, 7)

@DATASETS.register_module()
class FERPlusCondensed(BaseCondensedDataset):
    CONTEMPT = True
    
@DATASETS.register_module()
class FERPlusNegative(BaseNegativeDataset):
    CLASSES = ['Fear','Disgust','Sadness', 'Anger', 'Surprise', 'Contempt']
    CONTEMPT = True
