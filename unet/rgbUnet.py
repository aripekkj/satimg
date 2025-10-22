# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:26:33 2025

@author: E1008409
"""


import os
import glob
import numpy as np
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
import rasterio as rio

from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras import backend as K


def jacard(y_true, y_pred):
    y_true_c = K.flatten(y_true)
    y_pred_c = K.flatten(y_pred)
    intersection = K.sum(y_true_c * y_pred_c)
    return (intersection + 1.0) / (K.sum(y_true_c) + K.sum(y_pred_c) - intersection + 1.0) 

def jacard_loss(y_true, y_pred):
    return -jacard(y_true,y_pred)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)



# dir
os.getcwd()
os.chdir('/mnt/d/users/e1008409/MK/biodiversea/A6/hanko')
# output dir
outdir = os.path.join(os.getcwd(), 'mml', 'prediction')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
# input
fp_patches = os.path.join(os.getcwd(),'mml/mml_patches_256/*.tif')
fp_masks = os.path.join(os.getcwd(), 'mml/mml_patches_256_masks/*.tif')
# list images
imgs = []
fp_masks = [f for f in glob.glob(fp_masks)]
# find images according to masks
for m in fp_masks:
    fn = os.path.basename(m).split('_mask')[0]
    img_fp = glob.glob(os.path.join(os.path.dirname(fp_patches), fn + '.tif'))
    imgs.append(img_fp[0])

# -------------------------------------------------------------------- #
# MinMaxScale images
def readAndMinMax(img_fp):
    img = Image.open(img_fp)
    img = np.array(img)    
    # scale
    scaler = MinMaxScaler()
    image = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    return image

images = []
for i in imgs:
    im = readAndMinMax(i)
    images.append(im)
image_dataset = np.array(images)
# masks
masks = []
for i in fp_masks:
    mask = cv2.imread(i)
    print(i)
    mask = mask[:,:,0] / 255
    mask = np.where(mask > 0, 1, 0)
   
    print(np.unique(mask))
    #plt.imshow(mask)
    masks.append(mask)    

patch_size = 256
n_class = 2

#Sanity check, view few images
image_number = random.randint(0, len(imgs))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(images[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(masks[image_number], (patch_size, patch_size)))
plt.show()
print(imgs[image_number])
# one hot encode
labels_cat = to_categorical(masks, num_classes=n_class)

# Split train test data
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)

# to tf.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# ------------------------------------------------------------------- # 
#%%
# Data augmentation
BATCH_SIZE = 8
STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE
AUTOTUNE = tf.data.AUTOTUNE
seed = (42,0)
IMG_SIZE = 256

# https://www.tensorflow.org/tutorials/images/segmentation
class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=seed):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

#    self.augment_inputs = tf.keras.layers.RandomRotation(0.1, fill_mode='reflect', seed=seed)
#    self.augment_labels = tf.keras.layers.RandomRotation(0.1, fill_mode='reflect', seed=seed)

#    self.augment_inputs = tf.keras.layers.RandomBrightness((-0.2,0.2), seed=seed)
#    self.augment_labels = tf.keras.layers.RandomBrightness(0, seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

# Build the input pipeline, applying the augmentation after batching the inputs
train_batches = (
    train_dataset
    .cache()
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_dataset.batch(BATCH_SIZE)


def augment(image, seed):
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  new_seed = tf.random.split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.image.flip_left_right(image)
  image = tf.clip_by_value(image, 0, 1)
  return image


def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
idx = 10
testimg = X_train[idx]
testmask = y_train[idx]
visualize(testimg, testmask[:,:,1])

aug = augment(testimg, (20,0))
visualize(testimg, aug)

bright = tf.image.adjust_brightness(testimg, 0.2)
visualize(testimg, bright)

flip = tf.image.flip_left_right(testimg)
flipm = tf.image.flip_left_right(testmask)
visualize(flip, flipm[:,:,1])



#%%
# ------------------------------------------------------------------- # 
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model
#%%
model = simple_unet_model(patch_size, patch_size, 3)

metrics=['accuracy', jacard_coef]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

# fit model
history = model.fit(X_train, y_train,
        batch_size=8,
        verbose=1,
        epochs=25,
        validation_data=(X_test, y_test),
        shuffle=False
    )
# model with data augmentation
model_history = model.fit(train_batches, epochs=50,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_batches)

history = model_history

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)


#Using built in keras function for IoU
from keras.metrics import MeanIoU
#n_classes = 6
IOU_keras = MeanIoU(num_classes=n_class)  
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#######################################################################
#Predict on a few images

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()

#####################################################################
# make file list
num = np.arange(7,11,1)
num2 = np.arange(2, 9, 1)
predfiles = []
for n in num:
    for nn in num2:
        p = glob.glob(os.path.join(os.getcwd(),'mml/mml_patches_256/*' + str(n) + '_' + str(nn) + '.tif'))
        predfiles = predfiles + p
save_pred =True
# read new images and predict
for f in predfiles:
    test_img = readAndMinMax(f)
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:].astype('byte')
    # save prediction
    if save_pred == True:
        with rio.open(f) as src:
            profile = src.profile
        predout = os.path.join(outdir, os.path.basename(f))
        profileout = profile.copy()
        profileout.update(count=1,
                          dtype='uint8')
        
        with rio.open(predout, 'w', **profileout) as dst:
            dst.write(predicted_img.astype(profileout['dtype']),1)
        
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    plt.show()


