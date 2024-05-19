import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
 
# Constants
FER_BASE_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']
 
DATASET_CLASSES = ['Neutral', 'Happiness', 'Sadness','Surprise', 'Fear','Disgust', 'Anger']
 
data_folder = 'data\\FERPlus\\Image'  # Set to the path where your images are stored
ann_file = 'data\\FERPlus\\EmoLabel\\test.txt'
 
 
with open(ann_file, 'r') as file:
    file_names = [line.split(",")[0] for line in file.readlines()]
 
 
# Read the CSV
df_misaligned = pd.read_csv('missaligned_FERPlus.csv')
 
# Function to show an image and wait for a key press
def display_image_with_labels(idx, row, folder):
    plt.figure(figsize=(5, 5))
    img_name = file_names[row['Index']]
    image_path = os.path.join(folder, img_name)
    if os.path.exists(image_path):
        image = Image.open(image_path)
        plt.imshow(image, cmap="gray")
        predicted_label = FER_BASE_CLASSES[row['Result']]
        true_label = FER_BASE_CLASSES[row['GT_Label']]
        plt.title(f"Name:{img_name},Index: {row['Index']}\nPredicted: {predicted_label}, True: {true_label}")
        plt.axis('off')
        plt.draw()
        plt.waitforbuttonpress()  # Wait for a key press to continue
        plt.close()  # Close the plot to show the next one
    else:
        print(f"Image not found: {image_path}")
 
# Iterate through each row and display the image

#df_misaligned["Result"] = pd.to_numeric(df_misaligned["Result"], downcast="int32")
df_misaligned= df_misaligned.astype({"Result": int})
for index, row in df_misaligned.iterrows():

    display_image_with_labels(index, row, data_folder)
 