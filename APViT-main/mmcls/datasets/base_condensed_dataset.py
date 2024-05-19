
import numpy as np
from .base_fer_dataset import BaseFerDataset


class BaseCondensedDataset(BaseFerDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    DATASET_CLASSES = ["Neutral", "Happiness", "Other"]
    CLASSES = DATASET_CLASSES


    def convert_gt_label(self, i:int):
        return i

    def load_annotations(self):
        with open(self.ann_file) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
 
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            gt_label = int(gt_label)
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

