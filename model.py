from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D , Conv2DTranspose, UpSampling2D,Cropping2D
from keras.optimizers import SGD
import tensorflow as tf




def encoder(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1),trainable=False))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))


    if weights_path:
        model.load_weights(weights_path)

    return model

def decoder():
	model=encoder(weights_path='vgg16_weights.h5')

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(512, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2DTranspose(256, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(256, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(256, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2DTranspose(128, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(128, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2DTranspose(64, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))
	model.add(Conv2DTranspose(3, (3, 3), activation="elu"))
	model.add(Cropping2D((1,1)))

	return model

def VGG16_classification(width,height,weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(height,width,3)))
    model.add(Conv2D(64, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1),trainable=False))
    model.add(Conv2D(128, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu",trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2),trainable=False))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model



def conv1d_model(width,height=224):
    model=Sequential()
    model.add(Conv2D(3,(1,height),padding='valid',activation='relu',input_shape=(int(width),height,3)))
    return model



