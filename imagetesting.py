# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:47:44 2024

@author: keegd
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd




import os

# Load your pre-trained model
model = load_model('###############')

# Directory containing the test images
test_dir = '#############################'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


image_names = []
predictions = []

# Loop over each image file
for img_name in image_files:
    # Load the image
    img_path = os.path.join(test_dir, img_name)
    img = image.load_img(img_path, target_size=(299, 299, 3))  # Adjust target_size if needed

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
   
    
    # Run the prediction
    prediction = model.predict(img_array)
    image_names.append(img_name)
    predictions.append(prediction[0][0])
    
    # Print the result
    print(f"Image: {img_name}, Prediction: {prediction}")
predictiondf = pd.DataFrame({
    'Image Name': image_names,
    'Prediction': predictions
    })

#%%
#Using a cut off of 0.5, create a predicted bleach classification column
predictiondf['bleached'] = (predictiondf['Prediction'] < 0.5).astype(int)


#%%
#this section of the code is if you want to optimize your cut off point 

#hard coding the actual classification of the images 
predictiondf['actual'] = [1, 0, 0, 1, 1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,0,0,0,1,0,1,1]
#
# KS statistic for finding optimal cut off

# Extract the relevant columns
y_true = predictiondf['actual']  # Actual binary labels
y_pred_prob = predictiondf['Prediction']  # Predicted probabilities

y_pred_prob_0 = y_pred_prob[y_true == 0]
y_pred_prob_1 = y_pred_prob[y_true == 1]

# Sort the predicted probabilities
y_pred_prob_sorted = np.sort(y_pred_prob)

# Compute the cumulative distribution functions (CDFs) for each class
cdf_0 = np.searchsorted(np.sort(y_pred_prob_0), y_pred_prob_sorted, side='right') / len(y_pred_prob_0)
cdf_1 = np.searchsorted(np.sort(y_pred_prob_1), y_pred_prob_sorted, side='right') / len(y_pred_prob_1)

# Compute the KS statistic: the maximum difference between the two CDFs
ks_statistic = np.abs(cdf_0 - cdf_1)
optimal_idx = np.argmax(ks_statistic)
optimal_threshold = y_pred_prob_sorted[optimal_idx]

# Print the optimal threshold
print(f"Optimal threshold (KS statistic): {optimal_threshold}")
#%%

print(predictiondf)