import pandas as pd
import csv

# FER PLUS CLASSES
#['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# [Three Classes]
# ["Neutral", "Happiness", "Other"]
db_path = './data/FERPlus/EmoLabel/train.txt' #postavi path

'''Condensed emotions'''

'''AffectNet'''
# label_map_surprise_pos  ={0:2, 1:1,2:0,3:1,4:0,5:0,6:0}
# label_map_surprise_neg  ={0:2, 1:1,2:0,3:0,4:0,5:0,6:0}

'''RAF-DB'''
label_map_surprise_neg={0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:2, 7:2} # ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

label_map = label_map_surprise_neg

df = pd.read_csv(db_path, sep=',', header=None, names=['Image', 'Label'])
print("Broj razlicitih labela prije transformacije: ",df['Label'].value_counts().count)
print()
print("Broj razlicitih slika prije transformacije: ",df['Image'].value_counts())

df['Label'] = df['Label'].map(label_map)

print("Broj razlicitih labela NAKON transformacije: ",df['Label'].value_counts().count)
print()
print("Broj razlicitih slika NAKON transformacije: ",df['Image'].value_counts())


df.to_csv('./data/FERPlus/EmoLabel/train_big_three.txt',sep=" ", index=False, header=None) #

# '''Negative emotions'''

# df = pd.read_csv(db_path, sep='\\s+', header=None, names=['Image', 'Label'])

# # affecnet_neg_labels = [2,4,5,6,3]
# rafdb_neg_labels = [2, 3, 5, 6, 1, 7]

# neg_labels = rafdb_neg_labels

# df = df[df['Label'].isin(neg_labels)]

# '''
# Fear -> 0
# Disgust -> 1
# Sadness -> 2
# Anger -> 3
# Surprise -> 4
# '''

# '''AffectNet'''
# l_map_no_surprise = {4: 0, 5:1, 2:2, 6:3}
# l_map_with_surprise = {4: 0, 5:1, 2:2, 6:3, 3:4}

# '''RAF-DB'''
# l_map_with_surprise = {2: 0, 3:1, 5:2, 6:3, 1:4, 7:5}

# l_map = l_map_with_surprise

# df['Label'] = df['Label'].map(l_map)

# # Count the occurrences of each class
# val_count = df['Label'].value_counts()
# print(val_count)

# df.to_csv('data/RAF-DB/basic/EmoLabel/val_negative_emotions.txt', index=False, header=None) ##

 