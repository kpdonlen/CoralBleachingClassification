# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:23:03 2024

@author: keegd
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np



# Directory paths
data_dir = '#############'


# Create ImageDataGenerator with validation split
#when pulling the image data, this applies a variety of transformations to the images 
#to help generalize the training data of the model.
datagen = ImageDataGenerator(
    validation_split=0.1,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Load training data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#this sets how many steps the model should take during the fitting process before it finishes an epoch(datacycle)
steps_per_epoch = len(train_generator)


#this initializes the cNN we are using, "ResNet101", with the weights from the database ImageNet
base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(299, 299, 3))



#applying slight regularization to help with overfitting
l2_regularization = 0.001  
#base nn training methods, 1024 nodes with relu activation and sigmoid for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(.4)(x)
predictions = Dense(1, activation='sigmoid',kernel_regularizer=l2(l2_regularization))(x)

#initializes the resnet model with the specific training parameters
model = Model(inputs=base_model.input, outputs=predictions)

#freezes all nn layers
for layer in base_model.layers:
    layer.trainable = False
#unfreezes the top 2 layers for their weights to be retrained
for layer in base_model.layers[-2:]:
    layer.trainable = True

#compiles the model to be trained
model.compile(optimizer=Adam(learning_rate=.001), loss='binary_crossentropy', metrics=['accuracy'])


#sets early stopping parameters and defines the change of the learning rate during plateus
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

#fits and trains the model with previous parameters
model.fit(train_generator, epochs=100, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
    validation_steps=len(validation_generator), callbacks=[early_stopping,reduce_lr])



#%%

#this chunk of code is for fine tuning after initial fitting, found unneccesary 

#for layer in base_model.layers:
  #  layer.trainable = False

#for layer in base_model.layers[-4:]:
#    layer.trainable = True



#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(train_generator, epochs=100 , steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
    #validation_steps=len(validation_generator), callbacks = [early_stopping,reduce_lr])







#%%
model.save('#############')




#%%%

