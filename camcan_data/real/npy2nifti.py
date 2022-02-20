import numpy as np
import nibabel
import os
from glob import glob

files = glob(os.getcwd() + '/*.npy')

print(f'Converting {len(files)} files from npy to nii.gz')

for file in files:
    print(file)
    path, filename = os.path.split(file)
    im = np.load(file)
    imn = nibabel.Nifti1Image(im, np.eye(4))
    nibabel.save(imn, os.path.join(path, filename[:-3] + 'nii.gz'))
