# RLPP (**R**einforcement **L**earning **P**oint **P**rocess) Framework

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Usage](#usage)

# Overview
This repository provides the source code for training transregional spike prediction models with behavioral reinforcement under the RLPP framework, as described in the manuscript, [*A generative spike prediction model using behavioral reinforcement for re-establishing neural functional connectivity*](https://www.nature.com/articles/s43588-025-00915-5). The project is initially developed by MATLAB, and a Python version is also provided. 


# Repo Contents

- [data](./data): Preprocessed Data
- [decoding](./decoding): Movement decoders from M1 spike trains to behavioral labels.
- [draw_figures](./draw_figures): Scripts to plot results
- [model](./model): Transregional spike prediction models.
- [preprocess](./preprocess): Scripts for preprocessing raw spiking data (Neural selection, movement decoder training, *etc.*)
- [results](./results): training results (the folder will be automatically created after the model training)
- [trained_results](./trained_results/): Trained model weights and statistical results to reproduce the paper figures
- [training](./training): Training scripts for spike prediction models.
- [utils](./utils): Supporting functions.
- [Python_Ver](./Python_Ver/): Python version of this project


# System Requirements

## Hardware requirements
`RLPP` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software Requirements

### OS Requirements

The package is tested on *Win10* operating systems. The package should also be compatible with Mac and Linux operating systems.

### Dependencies

Before setting up the `RLPP` package, users should have `MATLAB R2021b` or higher, and the following toolboxes to run the code: 

- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) 
- [Statistics and Machine Learning Toolbox](https://www.mathworks.com/products/statistics.html)
- [Signal Processing Toolbox](https://www.mathworks.com/products/signal.html) 
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)
- [MATLAB Parallel Server](https://www.mathworks.com/products/matlab-parallel-server.html)

The python version requires Python 3.9 or higher. Detailed user guide is given below.

# Usage

## Simulation Demo
- To quickly obtain an intuitive understanding on the `RLPP` framework, you can run `main_simu.m` as a quickstart demo. 
    - Run `main_simu.m` to start data synthesis and model training
    - The results can be found in `/results/Simulations_RL_1.mat`. 
    - It may need several minutes to finish training. The result figures will be shown after the training.

## Reproducing results
- To reproduce the figures in the paper, please run the following scripts:

    - **Figure 2** `Fig2_NeuralModulation.m`: Individual neural modulation to the movements of neural recordings, RLPP predictions, and SLPP predictions; the differences in modulation between movements
    - **Figure 3** `Fig3_TimeDomain_And_SuccessRate.m`: Illustrating the model predictions in the time domain and comparing the statistical performance of generated neural activities
    - **Figure 4 and Supplementary Figure 4** `Fig4_ChangeDecoder_Supp_Fig4.m`: RLPP performance under different decoder settings
    - **Figure 5** `Fig5_Information.m`: Results of information analysis for spike prediction models
    - **Supplementary Figure 5** `Supp_Fig5_SpikePatterns.m`: Visualizing patterns of M1 spike ensembles using t-SNE
- Outputs may slightly different from the figures in the paper due to the randomly in spike generation, which do not change the main points of the paper.


## Model training
- You can train the transregional spike prediction models from scratch. The data from `Rat01` and `Rat02` in the paper are provided as examples.
    - Basic model training:
        - Run `main_real.m` to train models on the real recorded mPFC data by behavioral rewards
        - The trained results will be found in `results/`. The results can then by visualized by `Fig*.m` or `showSimuResults.m` with minor adaptions to data loading part of the scripts
    - Extending `RLPP` to different decoder settings
        - Run `main_mannualDecoder.m` to train models using the recorded mPFC spike trains and a manually designed decoder
        - Run `main_crossSubjectDecoder.m` to train models using the recorded mPFC spike of Rat02 and the movement decoder trained on Rat04 

## Implementing to new datasets
- To implement the `RLPP` framework on new datasets, you shall first preprocess the raw neural recordings into a similar format as `data/rat01.mat` and `data/rat02.mat`
- An example is given in `PreprocessRawData.m`, which corresponds to the M1 neuron selection and movement decoder training procedure described in the paper.

## Run with Python
- For Python users, no MATLAB or toolboxes are required for running the code. Please follow the steps below:
``` shell
# Create a virtual environment if needed
...

# copy data files to the python directory
# for Linux/MacOS
cp -r data trained_results Python_Ver/  
# for Windows
xcopy "data" "Python_Ver\data" /E /I
xcopy "trained_results" "Python_Ver\trained_results" /E /I 

# install packages
pip install -r Python_Ver/requirements.txt

# try the simulation demo:
python Python_Ver/main_simu.py  # Trained results will be saved in ./Python_Ver/results
# draw Fig. 3 
python Python_Ver/Fig3_TimeDomain_And_SuccessRate.py
# train from scratch
python Python_Ver/main_real.py
# ...(other functions)

```

# License

This project is covered under the **Apache 2.0 License**.
