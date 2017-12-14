
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2 , h5py
from model import *
from preprocess import *


bp=1 #balancing parameter
ar=0.8 #desired aspect ratio
width=224
height=224
depth=3	
batch_size=16


def cum_normalization(A_map,Ar):
	retargeted_width = int(ar*width)
	shift_map=np.ones((width,height,depth))
	cum_attention_map = np.cumsum(A_map,axis=0)
	cum_image=np.cumsum(Ar,axis=0)

	shift_map_y1=np.ones((retargeted_width,height,depth),dtype=np.float32)
	shift_map_y1=cum_attention_map/cum_image[:retargeted_width,:,:]

	shift_map_y2=np.ones((width-retargeted_width,height,depth),dtype=np.float32)
	last_col_vector=np.expand_dims(cum_attention_map[-1,:,:], axis=0)
	extended_attention_map=np.repeat(last_col_vector,45,axis=0)	
	shift_map_y2=extended_attention_map/cum_image[retargeted_width:,:,:]

	# shift_map=np.concatenate((shift_map_y1,shift_map_y2),axis=0)

	shift_map=cum_attention_map/cum_image
	return shift_map*(width-retargeted_width)

def attentionmap(img):
	ret_width = ar*width
	Ar=cv2.resize(img,(height,int(ret_width))) #Resize after attention map
	conv1dmodel=conv1d_model(ret_width,height)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	conv1dmodel.compile(optimizer=sgd, loss='categorical_crossentropy')
	img = np.expand_dims(Ar, axis=0)
	col_vector=conv1dmodel.predict(img)
	col_vector = col_vector.squeeze(axis=0)
	A1D = np.repeat(col_vector,height,axis=1) #duplicate coloumn vector
	A_Map=A1D+Ar
	# print A1D.shape, Ar.shape,A_Map.shape shape=(180,224,3)
	return A_Map , Ar


def attention_model():
	model= decoder(encoder('vgg16_weights.h5'))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print 'Encoder-Decoder Model Compiled'
	return model

def classification_model(w,h):
	model= VGG16_classification(w,h)
	sgd = SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print 'Classification Model Compiled'
	return model

def train_classification_model():
	model=classification_model(int(ar*width),height)
	X,Y=load_VOC_data()
	nb_images=len(X)
	for i in range(0,nb_images,16):
		X_batch,Y_batch=generate_batch(X[i:i+16],Y[i:i+16],retargeted_width,height)
		model.train_on_batch(X_batch,Y_batch)
	return 


if __name__=='__main__':
	retargeted_width = int(ar*width)
	train_classification_model()
	# model=encoder(weights_path='vgg16_weights.h5')
	# sgd = SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(optimizer=sgd, loss='categorical_crossentropy')
	# for i,layer in enumerate(model.layers):
		# print layer ,i
		# weights=model.get_weights()[i]

	# print len(model.get_weights())
	# model=classification_vgg_top(retargeted_width,height)
	# classification_model=vgg_top(retargeted_width,height)
	# for layer in model.layers:
		# print layer.name
	# for img in images:
		
	# 	im = np.expand_dims(im, axis=0)
	# 	im= model.predict(im)
	# 	im = im.squeeze(axis=0)
	# 	A_map , Ar=attentionmap(im)
	# 	shift_map=cum_normalization(A_map,Ar)
	# 	warped_image=cv2.resize(shift_map,(retargeted_width,height))

		
		# classification_model



		# out=warped_image
		# cv2.imwrite('color_img.jpg', out)



