# Spatial Transformer Network for German Traffic Sign Recognition Benchmark

## Overview
The German Traffic Sign Recognition Benchmark (GTSRB)(https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) is a multi-class image classification challenge. The goal of GTSRB is to get researchers in the field of Computer Vision to a common domain, where they could create models to recognize traffic signs and test it on a common dataset.

## Main Approach: 
The most prominent part of my final architecture is the Spatial Transformer Network (STN(, which was first introduced by Google DeepMind. 
### Experiment
Used two STN units with one batch normalization layer. Trained the 
model for 17 epochs and achieved an accuracy of 98.939. • Experiment 2 
Increased the number of STN units to 3 and increased the batch 
normalization layers. Trained the model for 20 epochs and achieved an 
accuracy of 99.255.

Detail explanation is  in **Report.pdf**.
