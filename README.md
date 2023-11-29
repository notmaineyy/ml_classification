# Enhancing Multi-class Image Classification using Data Augmentation

This project is to address the problem of having difficulties classifying items, due to the imbalanced dataset, with low data count for certain classes.
Dataset is not provided in this repo, but Python files can be reused for any dataset.

This was the project methodology:<br>
<img src="https://github.com/notmaineyy/ml_classification/assets/81574037/cca0f06b-ba6c-495a-a13d-9a2271c9f237" width="450"/>

## Data Augmentation Methods Experimented:
1. Traditional Methods<br>
   a. Flipping<br>
   b. Contrasting<br>
   c. Brightening<br>
   d. Sharpening<br>
2. Modern Methods<br>
   a. Generative Adverserial Network<br>
   b. Generative AI + Image Processing

After experimenting with various methods, the best augmentation methods were combined to increase the quantity of training and validation data. 

The best augmentation methods for this use case was using Generative AI to generate new images, processing the images to make it look similar to the original data, and applying traditional augmentation methods (flipping, contrasting and sharpening). 

The data distribution of the original dataset (before augmentation) and final dataset (after augmentation) can be seen here:<br>
<img src="https://github.com/notmaineyy/ml_classification/assets/81574037/5982aedb-82f9-4bc1-99ed-015cac83fab3" width="450"/>

The final outcome is the following:<br>
<img src="https://github.com/notmaineyy/ml_classification/assets/81574037/6a490a23-9c8f-47e8-a9c9-6c2fb42852d4" width="450"/>



