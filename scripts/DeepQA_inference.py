'''
Applies DeepQA model to images from a given image folder. Results are saved as a csv file.
'''
import enum
import os, argparse
import pandas as pd
import numpy as np
import scipy, pickle
from tqdm import tqdm

from os.path import join 
import tensorflow as tf
import tensorflow.keras.layers as layers
print('TF', tf.__version__)

import tensorflow_probability as tfp
import PIL, PIL.ImageOps

from pathlib import Path
home = str(Path.home())
print(home)

####################################
#       command line arguments     #
####################################
parser = argparse.ArgumentParser()
parser.add = parser.add_argument

parser.add('--image_dir', type=str, default='git/MRI-GAN-QA/3DStyleGAN/GM_png/032', help='folder with input images')
parser.add('--image_type', type=str, default='png', help='png jpg etc')
parser.add('--model_dir', type=str, default='git/MRI-GAN-QA/results')
parser.add('--model', type=str, default='rating', help='detection or rating')
parser.add('--outfile', type=str, default='git/MRI-GAN-QA/results/DeepQA_rating_results.csv', help='path with output filename')
parser.add('--resize', type=bool, default=True, help='resize images to 299x299')

args = parser.parse_args()

image_dir = args.image_dir if os.path.isabs(args.image_dir) else join(home, args.image_dir)
outfile = args.outfile if os.path.isabs(args.outfile) else join(home, args.outfile)
model_dir = args.model_dir if os.path.isabs(args.model_dir) else join(home, args.model_dir)

def load_images_from_folder(image_dir):
    '''Given a folder of images with subfolders, load images and crop/resize.
       Also returns a vector of class names (names of the respective subfolders)'''

    from glob import glob
    import re, PIL
    
    files = glob(join(image_dir, '**/*png'), recursive=True)
    assert len(files)>0, 'no image files found'
    n_images = len(files)
    print(f'loading {n_images} images')
    images = []
    
    for ix, file in enumerate(files):
        img = np.array(PIL.Image.open(file))
        img = img[75:210,45:400]  # cut away the empty area around the MRI StyleGAN images
        images.append(img)
        
    images = tf.stack(images, axis=0)
    if tf.rank(images)==3:
        images = tf.expand_dims(images, axis=-1) # add axis for channels if necessary
    if images.shape[-1]==1:
        images = tf.concat([images, images, images], axis=-1)

    # for VGG features, images need to be padded to (299,299)
    if args.resize: 
        images = tf.image.resize_with_pad(images, target_height=299, target_width=299)

    # scale
    images /= 127.5
    images -= 1

    # convert to tf dataset
    images = tf.data.Dataset.from_tensor_slices(images)
    # files = tf.data.Dataset.from_tensor_slices(files)
    # data = tf.data.Dataset.zip((images, files))
    return images, files # data


print(f'loading images and {args.model} model')
images, files = load_images_from_folder(image_dir)
images = images.batch(1)
if args.model == 'rating':
    model = tf.keras.models.load_model(join(model_dir,'rating_model'))
elif args.model == 'detection':
    model = tf.keras.models.load_model(join(model_dir,'detection_model_binary'))


print('applying model to images')
preds = []
for image in tqdm(images):
    pred = model(image)
    preds.append(tf.squeeze(pred))

print('saving to csv file')
folder = []
iteration = []
filename = []
for file in files: # split filepath into folder / iteration / filename
    tmp, fn = os.path.split(file)
    fold, it = os.path.split(tmp)
    folder.append(fold)
    iteration.append(it)
    filename.append(fn)

if args.model == 'rating':
    preds = np.array(preds) # these are the logits for each rating level
    rating = np.argmax(preds, axis=1) + 1
    df = pd.DataFrame({'iteration':iteration, 'filename':filename, 'logit_1':preds[:,0], \
        'logits_2':preds[:,1],'logits_3':preds[:,2],'logits_4':preds[:,3],'logits_5':preds[:,4], 'rating':rating})
    df.to_csv(outfile)
elif args.model == 'detection':
    df = []




