# MRI-GAN-QA

Data and code accompanying the paper:

[Treder, M. S., Codrai, R., & Tsvetanov, K. A. (2022). Quality assessment of anatomical MRI images from generative adversarial networks: Human assessment and image quality metrics. Journal of Neuroscience Methods, 374, 109579. https://doi.org/10.1016/J.JNEUMETH.2022.109579](https://doi.org/10.1016/j.jneumeth.2022.109579).

# Installation

Clone the repo and install the following packages:

- `tensorflow` (v2)
- `tensorflow_probability`
- `PIL`
- `opencv`
- `scipy`
- `numpy`
- `pandas`
- `openpyxl`
- `statsmodels`
- `scikit-learn`
- `jupyter`
- `matplotlib`
- `seaborn`
- `tqdm`

# Structure of the repo

The repository contains the following folders

- [`3DStyleGAN`](/3DStyleGAN): data and scripts on the control experiments using StyleGAN
- [`camcan_data`](/camcan_data): real and generated grey-matter (GM) density maps 
- [`experiment`](/experiment): data, images and scripts relating to the behavioral experiment
- [`figures`](/figures): figures of the analyses, mostly created with the Jupyter notebooks in this repo
- [`results`](/results): results of the main analysis of the behavioral data
- [`scripts`](/scripts): training and inference code for the Deep QA model and Jupyter notebooks with metrics and behavioral analyses
- [`videos`](/videos): videos illustrating the images generated by the WGAN

More details on these folders are provided below.

## 3DStyleGAN

The folder contains data, results and analysis scripts regarding the control analysis of grey matter density (GM) and T1 data using StyleGAN. It contains the following files and folders:

- [`analyze_2D_image_NIQE_BRISQUE.m`](3DStyleGAN/analyze_2D_image_NIQE_BRISQUE.m): MATLAB script that calculates NIQE and BRISQUE metrics on the StyleGAN data.
- [`analyze_2D_image_metrics_3DStyleGAN_GM.ipynb`](3DStyleGAN/analyze_2D_image_metrics_3DStyleGAN_GM.ipynb): image quality metrics and PCA on the GM data.
- [`analyze_2D_image_metrics_3DStyleGAN_T1.ipynb`](3DStyleGAN/analyze_2D_image_metrics_3DStyleGAN_T1.ipynb): image quality metrics and PCA on the T1 data.
- [`GM_png`](3DStyleGAN/GM_png): folder with png's for the GM data. The `real` subfolder contains the real data used to trained the StyleGAN, the numbered folders contain images exported at specific training iterations of the StyleGAN. For instance, `064` contains the generated GM images after 64,000 iterations.
- [`GM_repo`](3DStyleGAN/GM_repo): copy of the [3DStyleGAN repo](https://github.com/sh4174/3DStyleGAN). The [`run_training.py`](https://github.com/treder/MRI-GAN-QA/blob/main/3DStyleGAN/GM_repo/run_training.py#L40) has been adapted for GM data. The repo contains both the training scripts and the results.
- [`T1_png`](3DStyleGAN/T1_png): folder with png's for the T1 data.
- [`T1_repo`](3DStyleGAN/T1_repo): copy of the [3DStyleGAN repo](https://github.com/sh4174/3DStyleGAN) adapted for T1 data.
- [`T1_nifti`](3DStyleGAN/T1_nifti): T1-weighted images from the Cam-CAN dataset in Nifti format.
- [`T1_64x80x64_nifti`](3DStyleGAN/T1_64x80x64_nifti): version of the T1 at a size of 64x80x64.

## camcan_data

The folder [`camcan_data`](/camcan_data) contains grey-matter density maps for real images used to train the GAN. Additionally, it contains images generated by the GAN for five different iterations ranging from early to late stages of training. Each `.npy` file represents an individual MRI and is provided as a 3D Numpy array. To view them in a MRI viewer such as MRIcroGL they have to be converted to Nifti images using e.g. the [NiBabel](https://nipy.org/nibabel/gettingstarted.html) package.

## experiment 

The [`experiment`](/experiment) folder contains data, images and scripts relating to the behavioral experiment. 

- [`real`](experiment/real): 2D versions of the 3D GM Cam-CAN data. The images were generated by taking middle slice along x, y, and z directions and putting them next to each other. A subset of these was used in the behavioral experiment.
-  [`batch_344`](experiment/batch_344), [`batch_1055`](experiment/batch_1055), [`batch_7954`](experiment/batch_7954), [`batch_24440`](experiment/batch_24440),[`batch_60000`](experiment/batch_60000): 2D versions of the 3D images generated by the GAN. Showing middle slices for the respective batch numbers. A subset of these was used in the behavioral experiment.
- [`data`](experiment/data): raw behavioral data. For the analysis of the behavioral data refer to the Jupyter notebook [`analyze_psytoolkit_data.ipynb`](scripts/analyze_psytoolkit_data.ipynb).
- [`Psytoolkit`](/experiment/Psytoolkit): a download from the [PsyToolkit website](https://www.psytoolkit.org/) with all the Psytoolkit scripts. The Psytoolkit folder also contains an `images` subfolder with the subset of images that were used in the behavioral experiment. 

## scripts

Training and inference code for the Deep QA model and Jupyter notebooks showing the analyses of the metrics and behavioral data. It contains the following files:

- [`analyze_2D_image_metrics_detection_task.ipynb`](scripts/analyze_2D_image_metrics_detection_task.ipynb): image quality metrics and PCA on the images used in the behavioral detection task.
- [`analyze_2D_image_metrics_rating_task.ipynb`](scripts/analyze_2D_image_metrics_rating_task.ipynb): image quality metrics and PCA on the images used in the behavioral rating task.
- [`analyze_2D_image_NIQE_BRISQUE.m`](scripts/analyze_2D_image_NIQE_BRISQUE.m): MATLAB script for calculating NIQE and BRISQUE on the images.
- [`analyze_psytoolkit_data.ipynb`](scripts/analyze_psytoolkit_data.ipynb): analysis of the behavioral data.
- [`DeepQA_train.py`](scripts/DeepQA_train.py) and [`DeepQA_inference.py`](scripts/DeepQA_inference.py): training and inference code for the Deep QA model, explained next.

The Deep QA model is trained by running 

```
python DeepQA_train.py
```

It reads the behavioral data, extracts the empirical distibution of ratings for the images, and trains on the probabilistic labels. Set `train_detection_model = True` ([l.325](/scripts/DeepQA_train.py#L325)) to also train the detection model from scratch. You may need to adapt the file paths accordingly (ll. 23-26). The script can be taken as a starting point further improvements using data augmentation, merging with other datasets, or semi-supervised learning.

For inference, run `DeepQA_inference.py`. Given an input folder with images, the output is a `csv` file with the logits (for each of the five items on the rating scale) and the predicted rating (=arg max of the logits).

```
python DeepQA_inference.py --model_dir /path/to/model/ --image_dir /path/to/test/images
```

See [l.29-34](/scripts/DeepQA_inference.py#L29-L34) for additional command line parameters. 

