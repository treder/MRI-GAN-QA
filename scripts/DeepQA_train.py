'''
Trains the Deep QA model once one the whole rating task data and exports the model for 
inference
'''
import os
import pandas as pd
import numpy as np
import scipy, pickle

import tensorflow as tf
import tensorflow.keras.layers as layers
print('TF', tf.__version__)

import tensorflow_probability as tfp

import PIL, PIL.ImageOps
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from pathlib import Path
home = str(Path.home())
print(home)

basedir = os.path.join(home, 'git/MRI-GAN-QA/')
datadir = basedir + 'experiment/'
imagedir = basedir + 'experiment/Psytoolkit/'
resultdir = basedir + 'results/'

def clean_RT_data(df, RT_col, low_RT, high_RT, n_timeouts = None):
    '''Cleans RT data by removing too short RTs and removing participants with
    too many timeouts'''
    # time outs
    if n_timeouts is not None:
        timeout = df.groupby('participant')['timeout'].sum()
        timeout = timeout[timeout > n_timeouts]
        if timeout.shape[0] > 0:
            print(f'{timeout.shape[0]} participants have >{n_timeouts} timeouts, removing them')
            for ix in timeout.index:
                df = df[df['participant'] != ix]

    # check lower RT bound
    df_low = df[df[RT_col] <= low_RT]
    if df_low.shape[0] > 0:
        print(f'Removing {df_low.shape[0]} trials with RT <= {low_RT}')
        df = df[df[RT_col] > low_RT]

    # check high RT bound
    df_high = df[df[RT_col] >= high_RT]
    if df_high.shape[0] > 0:
        print(f'Removing {df_high.shape[0]} trials with RT >= {high_RT}')
        df = df[df[RT_col] < high_RT]

    return df

def load_images_from_dataframe(df, imagedir, resize=True):
    '''Given a dataframe of images, load images and crop/resize'''

    n_images = df.shape[0]
    print(f'Loading {n_images} images')
    images = np.zeros((n_images, 135, 355, 3))

    # go through each row and parse corresponding image
    for ix in range(n_images):
        im = np.array(PIL.Image.open(imagedir + 'images/' + df.iloc[ix, 0] + '.png'))[:,:,:3]
        im = im[75:210,45:400]  # cut away the empty area around the MRI
        images[ix, :, :, :] = im

    # for VGG features, images need to be padded to (299,299)
    if resize: 
        images = tf.image.resize_with_pad(images, target_height=299, target_width=299)
    else:
        images = tf.convert_to_tensor(images)

    # scale to the range [-1, 1]
    images /= 127.5
    images -= 1

    return images

def load_images_from_folder(imagedir, resize=True):
    '''Given a folder of images with subfolders, load images and crop/resize.
       Also returns a vector of class names (names of the respective subfolders)'''

    from glob import glob
    import re, PIL
    
    files = glob(imagedir + '**/*png')
    n_images = len(files)
    print(f'Loading {n_images} images')
    images = np.zeros((n_images, 135, 355, 3))
    classes = []
    
    # load images
    for ix, file in enumerate(files):
        im = np.array(PIL.Image.open(file))[:,:,:3]
        im = im[75:210,45:400]  # cut away the empty area around the MRI
        images[ix, :, :, :] = im
        # subfolder name is the class
        classes.append(re.findall(r'/(\w+)/[\w\d_]+\.png$',file)[0])
        

    # for VGG features, images need to be padded to (299,299)
    if resize: 
        images = tf.image.resize_with_pad(images, target_height=299, target_width=299)
    else:
        images = tf.convert_to_tensor(images)

    # scale to the range [-1, 1]
    images /= 127.5
    images -= 1

    return images, np.array(classes)

def get_detection_model(pretrained, detection_model_type, labels, pooling = None, input_shape=(299,299,3)):

    kwargs = {'include_top':False, 'weights':'imagenet', 'input_tensor':None,
             'input_shape':None, 'pooling':pooling, 'classes':1000}

    if pretrained == 'vgg19':
        base_model = tf.keras.applications.VGG19(**kwargs)
        base_model.trainable = False # freeze layers
    elif pretrained == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(**kwargs)
        base_model.trainable = False # freeze layers
    elif pretrained == 'densenet201':
        base_model = tf.keras.applications.DenseNet201(**kwargs)
        base_model.trainable = False # freeze layers
    elif pretrained == 'resnet101':
        base_model = tf.keras.applications.ResNet101(**kwargs)
        base_model.trainable = False # freeze layers
    elif pretrained == 'none':
        # build a trainable cnn instead
        base_model = tf.keras.models.Sequential([
            layers.Conv2D(8, kernel_size=3, strides=2),
            layers.Dropout(rate=0.1),
            layers.Conv2D(8, kernel_size=3, strides=2),
            layers.BatchNormalization(),
            # layers.MaxPool2D(pool_size=(2,2)),
            layers.Conv2D(16, kernel_size=3, strides=2),
            layers.Dropout(rate=0.1),
            layers.Conv2D(16, kernel_size=3, strides=2),
            layers.BatchNormalization(),
            # layers.MaxPool2D(pool_size=(2,2)),
            layers.Flatten()
        ])

    # prepare data augmentation   rotation = 0.1
    # data_augmentation = tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='nearest')

    # layers that transform the Imagenet features into features useful to MRI
    if (pretrained is not None) and (pooling == None):
        # we need a flatten layer first
        mri_features = tf.keras.models.Sequential([
                layers.Flatten(),
                layers.Dropout(0.1),
                layers.Dense(32),
                layers.LeakyReLU(alpha=0.1),
                layers.Dropout(0.1),
                layers.Dense(16),
                layers.LeakyReLU(alpha=0.1)
            ], name='mri_features')
    else:
        mri_features = tf.keras.models.Sequential([
                layers.Dense(32),
                layers.LeakyReLU(alpha=0.1),
                layers.Dropout(0.1),
                layers.Dense(16),
                layers.LeakyReLU(alpha=0.1)
            ], name='mri_features')


    # build model
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    # x = data_augmentation(x)
    x = base_model(x)
    mri_outputs = mri_features(x)

    if detection_model_type == 'binary':
        # Returns binary detection model that treats each all batches as 'fake' class
        detection_outputs = tf.keras.layers.Dense(1, name='class')(mri_outputs)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        labels = (labels==0).astype('int') # binarize class labels

    elif detection_model_type == 'multiclass':
        detection_outputs = tf.keras.layers.Dense(6, name='class')(mri_outputs)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else: raise ValueError(f'Unknown detection_model_type {detection_model_type}')

    detection_model = tf.keras.Model(inputs, detection_outputs, name=detection_model_type + '_detection_model')

    # Compile DETECTION model
    detection_model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_detection),
              loss=loss,
              metrics=['accuracy'])
    # detection_model.summary()
    return detection_model, base_model, loss, inputs, mri_features, mri_outputs, labels

def unfreeze_conv_layers(n, model):
    '''Unfreezes the last n conv layers and all following layers'''
    conv_ix = [ix for ix, layer in enumerate(model.layers) if 'conv' in layer.name]
    for layer in model.layers[conv_ix[-n]:]:
        layer.trainable = True

def freeze_conv_layers(n, model):
    '''Freezes the last n conv layers and all following layers'''
    conv_ix = [ix for ix, layer in enumerate(model.layers) if 'conv' in layer.name]
    for layer in model.layers[conv_ix[-n]:]:
        layer.trainable = False

def sample_targets(p_yx_distr):
    '''Samples target vector from an empirical TF distribution'''
    targets = np.zeros((p_yx_distr.shape[0],))
    for ix in range(targets.shape[0]):
        targets[ix] = p_yx_distr[ix].sample((1,)).numpy()
    return targets

def train_step(x, y, model, loss_fn):
    y0 = y - 1   # y ranges from 1 to 5, not 0 to 4
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y0, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y0, logits)
    return loss_value

# --- process RATING task labels ---

with open(resultdir + f'psytoolkit_all_participants26.pickle', 'rb') as f:
    df_rating = pickle.load(f)

# select only rating task
df_rating = df_rating[df_rating.task == 'RATING_TASK']

# clean RTs like in analyze_toolkit_data
df_rating = clean_RT_data(df_rating, 'rate_RT', low_RT=150, high_RT=10000)

# dummy code rating
df_rating = pd.concat((df_rating, pd.get_dummies(df_rating.rate, prefix='rate')), axis=1)
df_rating['batch'] = df_rating['batch'].replace({344:0, 1055:1, 7954:2, 24440:3, 60000:4, 'real':5}).astype('int')

# calculate empirical posterior pobabilities for each of the 30 images by averaging each rate across participants
p_yx = df_rating.groupby('tablerow').apply(lambda x:
                pd.Series({'batch':np.unique(x.batch)[0], 'rate':x.rate.mean(),
                          'rate_1':x['rate_1'].mean(),'rate_2':x['rate_2'].mean(),'rate_3':x['rate_3'].mean(),
                          'rate_4':x['rate_4'].mean(),'rate_5':x['rate_5'].mean()})).reset_index().copy()
print(p_yx.shape)
p_yx.batch = p_yx.batch.astype('int')

# for sampling: get empirical probability distribution object for each of the 30 rating images
p_yx_distr = []
for ix in range(p_yx.shape[0]):
    p_yx_distr.append(tfp.distributions.Empirical(df_rating[df_rating.tablerow==ix+1]['rate']))
p_yx_distr = np.array(p_yx_distr, dtype=np.object)

# Bayes maximum average accuracy for the model given that the labels are fuzzy
bayes_accuracy = p_yx.loc[:,'rate_1':'rate_5'].max(axis=1).mean()
print('Bayes accuracy limit:', bayes_accuracy)

# -- Load INDEPENDENT images and labels to train detection model ---
exclude_images_used_in_experiment = True

#if exclude_images_used_in_experiment:
print('Loading images NOT used in experiment')
independent_images, independent_labels = load_images_from_folder(datadir + 'images_excluding_images_used_in_experiment/')
independent_labels[independent_labels == 'real'] = 0
independent_labels[independent_labels == 'batch_344'] = 1
independent_labels[independent_labels == 'batch_1055'] = 2
independent_labels[independent_labels == 'batch_7954'] = 3
independent_labels[independent_labels == 'batch_24440'] = 4
independent_labels[independent_labels == 'batch_60000'] = 5
independent_labels = independent_labels.astype('int')
im_shape = independent_images.shape[1:]
n_detection_epochs_initial = 5
n_detection_epochs = 20

# -- Load DETECTION images and labels (for testing the detection model) ---
print('Loading images used in detection experiment')    
detection_table = pd.read_csv(datadir + 'Psytoolkit/main_experiment_table.txt', sep=' ', header=None, names=['image','button','batch']).reset_index(drop=True)
detection_table.loc[detection_table.batch == 0, 'batch'] = 6
detection_table['batch'] -= 1
# detection_table['batch'] = detection_table.batch.astype('category')

detection_experiment_images = load_images_from_dataframe(detection_table, imagedir)

# --- Load RATING images ---
rating_table = pd.read_csv(datadir + 'Psytoolkit/rating_table.txt', sep=' ', header=None, names=['image','batch']).reset_index(drop=True)
rating_table.loc[rating_table.batch == 0, 'batch'] = 6
rating_table['batch'] -= 1
rating_images = load_images_from_dataframe(rating_table, imagedir).numpy()
print('rating images:', rating_images.shape)

# Parameters
n_rating_epochs_initial = 20
n_rating_epochs = 200
n_runs = 100
n_splits = 5 # for xvalidation

lr_detection = 1e-4
lr_rating = 1e-4

# these keep the information we want to save
detection_hist = []
rating_hist = np.zeros((n_runs, n_splits, n_rating_epochs_initial+n_rating_epochs, 3)) # nepochs x loss/train acc/test acc
rating_pred = []

#pretrained = 'vgg19'
pretrained = 'inceptionv3'
#pretrained = 'resnet101'
#pretrained = 'densenet201'
#pretrained = 'none'

print(f'Pretrained model: {pretrained.upper()}')
detection_task = 'binary'
#detection_task = 'multiclass'

#######################
#   DETECTION MODEL   #
#######################
tf.keras.backend.clear_session()
# get detection model

train_detection_model = False
if train_detection_model:
    detection_model, base_model, loss, inputs, mri_features, mri_outputs, independent_labels =\
        get_detection_model(pretrained, detection_task, independent_labels, input_shape=im_shape)

    x_train= independent_images.numpy()
    y_train = independent_labels

    print(f'Training DETECTION model for {n_detection_epochs_initial} epochs with pretrained layers frozen')
    detection_model.fit(x_train, y_train, batch_size = 16, epochs = n_detection_epochs_initial, verbose = 0)

    print(f'Training DETECTION model for {n_detection_epochs} epochs with last pretrained layers unfrozen')
    unfreeze_conv_layers(2, base_model)
    detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_detection*0.1),loss=loss, metrics=['accuracy'])

    detection_model.fit(x_train, y_train, batch_size = 16, epochs = n_detection_epochs, verbose = 2)
    detection_model.save_weights(resultdir + 'detection_model_weights_' + detection_task)
    detection_model.save(resultdir + 'detection_model_' + detection_task)

#######################
#     RATING MODEL    #
#######################
print('Training RATE model')

# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
kfold = KFold(n_splits=n_splits, shuffle=True)

batch_label = p_yx.batch.astype('object')
batch_label[batch_label=='real'] = -1
batch_label = batch_label.astype('int')

# rating_targets = p_yx.iloc[:, -5:].to_numpy()
rating_targets = np.zeros((rating_table.shape[0], n_rating_epochs_initial+n_rating_epochs), dtype=np.int)
# targets range from 1 to 5 but we need 0 to 4 for classification
rating_targets -= 1

# sample target labels from empirical distribution
# rating_targets = sample_targets(rating_targets, p_yx_distr)
    # for ix in range(p_yx.shape[0]):
    #     rating_targets[ix, :] = p_yx_distr[ix].sample((n_rating_epochs_initial+n_rating_epochs,)).numpy()

# pred will keep the out-of-sample predictions, pre initialise with -1
rating_table['pred'] = -1
rating_table['rate_1'] = -1
rating_table['rate_2'] = -1
rating_table['rate_3'] = -1
rating_table['rate_4'] = -1
rating_table['rate_5'] = -1

detection_model, base_model, loss, inputs, mri_features, mri_outputs, independent_labels = \
    get_detection_model(pretrained, detection_task, independent_labels, input_shape=im_shape)


x_train = rating_images
gan_batch_train = rating_table.batch.to_numpy()

# create rating dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, np.arange(x_train.shape[0])))   # data and indices
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, p_yx_distr[train_index]))
train_dataset = train_dataset.shuffle(buffer_size=30).batch(batch_size=6)

# reload weights of detection model
tf.keras.backend.clear_session()
detection_model.load_weights(resultdir + 'detection_model_weights_' + detection_task)

# Replace last layer by rating layer
rating_outputs = tf.keras.layers.Dense(5, activation='linear', name='rating')(mri_outputs)

# freeze the MRI features initially
rating_model = tf.keras.Model(inputs, rating_outputs, name='rating_model')
mri_features.trainable = False

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rating)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Stage 1: freeze MRI features, only train final layer
# Custom training loop: we sample the targets in every batch/epoch
# from the conditional distribution p(y|x)
train_accs = []
test_accs = []
losses = []
for epoch in range(n_rating_epochs_initial):
    # Iterate over the batches of the dataset
    total_loss = 0
    for step, (x_batch_train, batch_indices) in enumerate(train_dataset):
        # step, (x_batch_train, batch_indices) = next(enumerate(train_dataset))
        y_batch_train = sample_targets(p_yx_distr[batch_indices.numpy()])
        loss_value = train_step(x_batch_train, y_batch_train, rating_model, loss_fn)
        total_loss += loss_value.numpy()

    losses.append(total_loss)
    # Train accuracy
    train_accs.append(train_acc_metric.result())
    train_acc_metric.reset_states()    # Reset training metrics at the end of each epoch

    if epoch % 4 == 0: print(f'{epoch} Loss: {total_loss:2.2f}, Train acc: {train_accs[-1]:2.2f}')


# Stage 2: unfreeze MRI features, continue training
print('Unfreeze MRI features')
mri_features.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rating*0.1)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_accs = []; test_accs = []; losses = []
for epoch in range(n_rating_epochs):
    total_loss = 0
    for step, (x_batch_train, batch_indices) in enumerate(train_dataset):
        y_batch_train = sample_targets(p_yx_distr[batch_indices.numpy()])
        loss_value = train_step(x_batch_train, y_batch_train, rating_model, loss_fn)
        total_loss += loss_value.numpy()

    losses.append(total_loss)
    # Train accuracy
    train_accs.append(train_acc_metric.result())
    train_acc_metric.reset_states()    # Reset training metrics at the end of each epoch

    if epoch % 4 == 0: print(f'{epoch} Loss: {total_loss:2.2f}, Train acc: {train_accs[-1]:2.2f}')

# Save model
rating_model.save_weights(resultdir + 'rating_model_weights')
rating_model.save(resultdir + 'rating_model')
