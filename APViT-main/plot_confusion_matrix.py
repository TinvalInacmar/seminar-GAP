import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from mmcls.datasets.base_fer_dataset import FER_BASE_CLASSES
import numpy as np
 
df_misaligned = pd.read_csv('all_results_FERPlus.csv')
 
predictions = df_misaligned['Result']
gt = df_misaligned['GT_Label']
 
class_indices = [i for i in range(len(FER_BASE_CLASSES))]
 
cm = confusion_matrix(gt, predictions, labels=class_indices, normalize='true').round(2)
 
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']) #['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'] 
disp.plot()
#plt.show()
plt.savefig('ensamble_confusion_matrix.png')
