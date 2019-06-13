from __future__ import print_function
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model

import numpy as np
import resnet
import os, cv2
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

checkpoint_folder = './checkpoints/resnet50_flower_finetuned_v2/'
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('./logs/resnet50_flower_finetuned_v2.csv')
checkpointer = ModelCheckpoint(checkpoint_folder+'resnet50-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5', verbose=1, save_best_only=True)

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

make_train_data('Daisy',TRAIN_FLOWER_DAISY_DIR)
make_train_data('Sunflower',TRAIN_FLOWER_SUNFLOWER_DIR)
make_train_data('Tulip',TRAIN_FLOWER_TULIP_DIR)
make_train_data('Dandelion',TRAIN_FLOWER_DANDI_DIR)
make_train_data('Rose',TRAIN_FLOWER_ROSE_DIR)
make_test_data('Daisy',TEST_FLOWER_DAISY_DIR)
make_test_data('Sunflower',TEST_FLOWER_SUNFLOWER_DIR)
make_test_data('Tulip',TEST_FLOWER_TULIP_DIR)
make_test_data('Dandelion',TEST_FLOWER_DANDI_DIR)
make_test_data('Rose',TEST_FLOWER_ROSE_DIR)

# reshape labels
le=LabelEncoder()
Y_train=le.fit_transform(Z_train)
Y_train=to_categorical(Y_train,5)
X_train=np.array(X_train)
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

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = ResNet50()
model.layers.pop()
for layer in model.layers:
    layer.trainable=True
last = model.layers[-1].output
x = Dense(nb_classes, activation="softmax")(last)
finetuned_model = Model(model.input, x)
finetuned_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
finetuned_model.summary()

if not data_augmentation:
    print('Not using data augmentation.')
    finetuned_model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    finetuned_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer])

finetuned_model.save(checkpoint_folder+'resnet_final.h5')
