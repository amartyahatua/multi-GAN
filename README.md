##All the results of this experiment is submitted at LDK 2021.

## Requirements
* **PyTorch r1.1.0**
* Python 3.5+
* CUDA 8.0+ (For GPU)

## File
* datafile.py: This file prepares the data for the array_gan.py file. This needs to run at first to generate postive, negative and data files.
* array_gan.py: This file contains all important functions to and implements a Multi-GAN based algorithm to perform claim verification
* dataset.py: This file loads the data to for the array_gan.py
* func.py: This file contains some utility functions for this project, which have been called multiple times
* tsne_plot.py: Plots all loss functions and generated data
* similarity_score.py: Calculate the similarity score (Cosine, Manhattan, Euclidean)

## Instructions to run the programs
* Set the count variable in datafile.py according to number of you data points need to be processed
* Use the following commands: 
* python datafile.py
* python array_gan.py