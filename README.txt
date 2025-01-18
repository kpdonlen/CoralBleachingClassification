I created this side project to classify images of coral as bleached or healthy. 

The coralid.py file is the code to train the initial CNN. It will require you to read a directory that contains the images you wish to train on. Within this directory, there will need to be two folders, one that has all of the bleached coral and the other that has all of the healthy coral. This is how the model determines the correct answer during training. It uses transfer learning courtesy of resnet151 trained on imagenet.

After you train a model and have saved it, you can then make predictions on images the model has not seen before using the imagetesting.py file

The coral_bleaching_classifierV9.h5 file is the final model I created. After training it on over 900 images, it had a final accuracy of about 93.3% on about 50 images of coral I found online. You can load it into the imagetesting.py file and immediately test it on images of coral.

Thank you for your interest in my project!
-Keegan Donlen
