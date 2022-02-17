# MRI-GAN-QA

Data and code accompanying the paper [INSERT PAPER LINK].

# Installation

Clone the repo and install the following packages:

- `tensorflow` (v2)
- `tensorflow_probability`
- `PIL`
- `opencv`
- `scipy`
- `numpy`
- `pandas`
- `statsmodels`
- `scikit-learn`
- `jupyter`
- `matplotlib`
- `seaborn`
- `tqdm`

### Structure of the repo

The repository contains the following folders

- [`camcan_data`](/camcan_data): real and generated grey-matter (GM) density maps 
- [`experiment`](/experiment): data, images and scripts relating to the behavioral experiment
- [`results`](/results): results of the main analysis of the behavioral data and the metrics

# CamCAN data

The folder [`camcan_data`](/camcan_data) contains grey-matter density maps for real images used to train the GAN. Additionally, it contains images generated by the GAN for five different iterations ranging from early to late stages of training. Each `.npy` file represents an individual MRI and is provided as a 3D Numpy array. To view them in a MRI viewer such as MRIcroGL they have to be converted to Nifti images using e.g. the [NiBabel](https://nipy.org/nibabel/gettingstarted.html) package.

# StyleGAN



# Deep QA model

