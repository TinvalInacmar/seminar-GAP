from .builder import DATASETS

from .base_fer_dataset import BaseFerDataset

@DATASETS.register_module()
class FER2013(BaseFerDataset):
    DATASET_CLASSES = ['Happiness', 'Fear', 'Sad','Surprise', 'Neutral', 'Disgust', 'Angry']
    CONVERT_TABLE = (4, 2, 3, 5, 6, 1, 0)
