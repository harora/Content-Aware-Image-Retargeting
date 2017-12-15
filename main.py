
import os
import h5py
import math

import cv2
import keras
import tensorflow as tf
import numpy as np

from keras import backend as K
from skimage import transform as trf
from model import *
from preprocess import *


bp=1 #balancing parameter
ar=0.8 #desired aspect ratio
width=224
height=224
depth=3	
batch_size=16
nb_classes=20
epsilon=1e-7
repeat_tensor=0 #Global declaration


def content_loss(y_true,y_pred):
	'''
	Content Loss
	Input : Actual values and predictions. (batch_size,nb_classes)
	Output: Log loss implemented in papeer
	'''
	losses = -y_pred*(tf.log(y_true+epsilon)) - (1-y_true)*(tf.log(1-y_pred+epsilon))
	return losses/(nb_classes*batch_size)


def cum_normalization(A_map,Ar=repeat_tensor):
	'''
	Perform Cumulative Normalization after Conv1D model.
	'''
	retargeted_width = int(ar*width)
	shift_map=x = tf.placeholder(tf.float32, shape=(height,retargeted_width,depth))

	cum_attention_map = np.cumsum(A_map,axis=2) #Cumulative sum about width axis
	cum_image=np.cumsum(Ar,axis=2)
	
	# Assuming Normalization is to be done image before it is resized 
	shift_map_y1=np.ones((retargeted_width,height,depth),dtype=np.float32)
	shift_map_y1=cum_attention_map/cum_image[:retargeted_width,:,:]

	shift_map_y2=np.ones((width-retargeted_width,height,depth),dtype=np.float32)
	last_col_vector=np.expand_dims(cum_attention_map[-1,:,:], axis=0)
	extended_attention_map=np.repeat(last_col_vector,45,axis=0)	
	shift_map_y2=extended_attention_map/cum_image[retargeted_width:,:,:]

	shift_map=np.concatenate((shift_map_y1,shift_map_y2),axis=0)

	#Dividing attention map over initial putput of decoder
	shift_map=tf.divide(tf.convert_to_tensor(cum_attention_map),tf.to_float(tf.convert_to_tensor(cum_image)))
	return shift_map*(width-retargeted_width)

def attentionmap(img):
	'''
	Attention Map to be used while testing
	Input : Np.array(2 dim)
	Output : Attention Map
	'''
	ret_width = ar*width
	Ar=cv2.resize(img,(height,int(ret_width))) #Resize after attention map
	img = np.expand_dims(Ar, axis=0)

	col_vector=conv1dmodel.predict(img) # Prediction on 1D model

	col_vector = col_vector.squeeze(axis=0)
	A1D = np.repeat(col_vector,height,axis=1) #duplicate coloumn vector
	A_Map=A1D+Ar
	return A_Map , Ar

def classification_model(w,h):
	'''
	Comiles the classification_model. 
	This is trained using categorical_crossentropy(Label are binary arrays) 
	'''
	model= VGG16_classification(w,h)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print 'Classification Model Compiled'
	return model

def train_classification_model():
	'''
	Train the classification Model 
	Sets the trained weights at all layers except fully convolutional layers
	The weights are saved and later used for training complete model.

	'''
	model=classification_model(int(ar*width),height)
	
	f = h5py.File('vgg16_weightswithtop.h5')
	layer_with_weights=[1,3,4,6,8,9,11,13,15,16,18,20,22,23,25,27,29,30] #convolution and max pooling layers
	for idx,l in enumerate(layer_with_weights):
		if(idx<18): 
			g= f[f.attrs.values()[0][idx]]
			weights = [g[g.attrs.values()[0][i]] for i in range(len(g))] #Setting weights for intermediate layers
			model.layers[l].set_weights(weights)
		print idx,l
	
	X,Y=load_VOC_data()
	nb_images=len(X)
	
	for ep in range(200):
		for idx,i in enumerate(range(0,nb_images,16)):
			print 'Epoch: ',ep,' Batch: ',idx+1
			X_batch,Y_batch=generate_batch(X[i:i+16],Y[i:i+16],retargeted_width,height) #Fetching a batch
			loss=model.train_on_batch(X_batch,Y_batch)			
			print 'Loss: ',loss,' Accuracy: '
		model.save_weights('weights'+str(ep)+'.h5')
	return 

def resize(image):
    import tensorflow as tf  #Workaround since tf dosen't exist in Lambda function
    resized_image = tf.image.resize_images(image, (height,int(ar*width)))
    return resized_image

def repeat(col_vector,axis=0):
	'''
	Duplication of tensor with a specific axis
	'''
	return K.repeat_elements(col_vector,height,axis=1)
	
def merge(r1):
	'''
	Addition of two tensors
	'''
	return tf.add(r1,repeat_tensor)

def warp(im):
	'''
	Warping Function. 
	Input : A tensor of 4 dimensions (None,h,w,nb_channels)
	'''
	sess= tf.Session() 
	with sess.as_default(): #Conversion from tensor to array
		im=im.eval()

	afine_tf = trf.AffineTransform(shear=0.4, translation=[(1-ar) * width, 0]) #Linear Mapping using Affine Transform
	modified = trf.warp(im, inverse_map=afine_tf)
	
	return tf.convert_to_tensor(modified)


def generate_model():
	'''
	End to End model for training. 
	The model has
	- Encoder model , Decoder Layer , Resize Layer, Conv1D layer, Duplication layer
	- Merge Layer , Cumulative Normalization layer , Warping layer , VGG Classification model
	'''
	retargeted_width=int(ar*width)

	model=encoder()
	model.add(decoder())

	model.add(Lambda(resize, input_shape=(height,width,depth), output_shape=(height,retargeted_width,depth),name='resize1'))
	
	model.add(conv1d_model(retargeted_width)) #Conv1D layer
	
	model.add(Lambda(repeat,input_shape=(1,retargeted_width,depth),output_shape=(height,retargeted_width,depth),name='repeat1')) #Duplication layer
	repeat_tensor=model.layers[32].output
	
	model.add(Lambda(merge,input_shape=(height,retargeted_width,depth),output_shape=(height,retargeted_width,depth)))
	
	model.add(Lambda(cum_normalization,input_shape=(height,retargeted_width,depth))) #Cumulative Normalization
	
	# model.add(Lambda(resize, output_shape=(height,retargeted_width,depth)))
	model.add(Lambda(warp, output_shape=(height,retargeted_width,depth))) # Warping layer
	
	model.add(VGG16_classification(retargeted_width,height))

	return model

def train():
	model=generate_model()
	f = h5py.File('vgg16_weightswithtop.h5')
	
	layer_with_weights=[1,3,4,6,8,9,11,13,15,16,18,20,22,23,25,27,29,30] #convolution and max pooling layers
	for idx,l in enumerate(layer_with_weights):
		if(idx<18):
			g= f[f.attrs.values()[0][idx]]
			weights = [g[g.attrs.values()[0][i]] for i in range(len(g))]
			model.layers[l].set_weights(weights)


if __name__=='__main__':
	retargeted_width = int(ar*width)
	train()
	



