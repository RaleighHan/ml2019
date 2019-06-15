from __future__ import print_function
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model, load_model

import numpy as np
import resnet
import os, cv2
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_VISIBLE_DEVICES"] = ""

batch_size = 32
nb_classes = 5
nb_epoch = 200
data_augmentation = True

# import data
X_train=[]
Z_train=[]
X_test=[]
Z_test=[]
IMG_SIZE=224
TRAIN_FLOWER_DAISY_DIR='../data/train_test/train/daisy'
TRAIN_FLOWER_SUNFLOWER_DIR='../data/train_test/train/sunflower'
TRAIN_FLOWER_TULIP_DIR='../data/train_test/train/tulip'
TRAIN_FLOWER_DANDI_DIR='../data/train_test/train/dandelion'
TRAIN_FLOWER_ROSE_DIR='../data/train_test/train/rose'
TEST_FLOWER_DAISY_DIR='../data/train_test/test/daisy'
TEST_FLOWER_SUNFLOWER_DIR='../data/train_test/test/sunflower'
TEST_FLOWER_TULIP_DIR='../data/train_test/test/tulip'
TEST_FLOWER_DANDI_DIR='../data/train_test/test/dandelion'
TEST_FLOWER_ROSE_DIR='../data/train_test/test/rose'

def assign_label(img, flower_type):
    return flower_type

def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X_train.append(np.array(img))
        Z_train.append(str(label))

def make_test_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X_test.append(np.array(img))
        Z_test.append(str(label))

make_test_data('Daisy',TEST_FLOWER_DAISY_DIR)
make_test_data('Sunflower',TEST_FLOWER_SUNFLOWER_DIR)
make_test_data('Tulip',TEST_FLOWER_TULIP_DIR)
make_test_data('Dandelion',TEST_FLOWER_DANDI_DIR)
make_test_data('Rose',TEST_FLOWER_ROSE_DIR)

# reshape labels
le=LabelEncoder()
Y_test=le.fit_transform(Z_test)
Y_test=to_categorical(Y_test,5)
X_test=np.array(X_test)

# train/test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# input image dimensions
img_rows, img_cols = IMG_SIZE, IMG_SIZE
# The images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_test, axis=0)
X_test -= mean_image
X_test /= 128.

print(X_test.shape, Y_test.shape)

finetuned_model = load_model('./checkpoints/resnet50_flower_finetuned/resnet50-18-0.3222-0.9038.h5')
finetuned_model.summary()

loss, acc = finetuned_model.evaluate(X_test, Y_test)

print('Accuracy: %.6f' % acc)


