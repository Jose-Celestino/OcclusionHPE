# Occlusion_HPE

This is a repository for code regarding head pose estimation for occluded face images and generation of synthetic occlusion in face images.
This repository contains Python, iPython and Matlab code.

## Trained Models for HPE

Go to the models folder and download the Google Drive zip file. This file contains all necessary trained models for HPE with occlusions and for further personalized training. 
Extract the models to the models folder in this repository.

## Datasets for Training and Testing

The datasets folder in this repository contains Google Drive links for three datasets (non-occluded and occluded versions) and respective annotations: 300W-LP, BIWI and AFLW2000 datasets. Extract the datasets to the datasets folder. Additionally, there is a link for occlusion dimension/level datasets. 

## Code

# Training

To train of your own model, run Train_Latent.py. 

# Testing Individual Images

For individual image inference use the iPynthon file Test_Latent_Inference.ipynb.
Preferably, use the Latent_model_0,999.pkl model for best yaw estimation in inference.

# Testing Datasets

For dataset inference use the iPynthon file Test_Latent_Datasets.ipynb.
Preferably, use the Latent_model_0,999.pkl model for best yaw estimation in inference.
In datasets.py has customized classes for each dataset, easy to adapt and use in training and testing. The model is defined in LatentNet.py. 

# Testing Occlusion Levels

For occlusion level inference use the iPynthon file Test_Occlusion_Dimension_Inference.ipynb and the occlusion level datasets.

There are two Matlab files in the synthetic occlusion folder for the generation of synthetic occlusion in images and datasets.
Kinect_Occlusion_Recorder.m allows to record the occluded RGB and depth data necessary for the procedure in a .mat file. 
Once the .mat file is created, run the Synthetic_Occlusion_Generator with the desired dataset to occlude.
