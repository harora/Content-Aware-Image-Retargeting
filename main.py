
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2 , h5py , math,keras
from keras import backend as K
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


def content_loss(y_true,y_pred):
	losses = -y_pred*(tf.log(y_true+epsilon)) - (1-y_true)*(tf.log(1-y_pred+epsilon))
	return losses/(nb_classes*batch_size)


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
	img = np.expand_dims(Ar, axis=0)
	col_vector=conv1dmodel.predict(img)
	col_vector = col_vector.squeeze(axis=0)
	A1D = np.repeat(col_vector,height,axis=1) #duplicate coloumn vector
	A_Map=A1D+Ar
	return A_Map , Ar


def attention_model():
	model=encoder()
	# model.add(decoder())
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss=content_loss)
	print 'Encoder-Decoder Model Compiled'
	return model

def classification_model(w,h):
	model= VGG16_classification(w,h)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print 'Classification Model Compiled'
	return model

def train_classification_model():
	model=classification_model(int(ar*width),height)
	f = h5py.File('vgg16_weightswithtop.h5')
	layer_with_weights=[1,3,4,6,8,9,11,13,15,16,18,20,22,23,25,27,29,30] #convolution and max pooling layers
	for idx,l in enumerate(layer_with_weights):
		if(idx<18):
			g= f[f.attrs.values()[0][idx]]
			weights = [g[g.attrs.values()[0][i]] for i in range(len(g))]
			model.layers[l].set_weights(weights)
		print idx,l
	X,Y=load_VOC_data()
	nb_images=len(X)
	for ep in range(2):
		for idx,i in enumerate(range(0,nb_images,16)):
			print 'Epoch: ',ep,' Batch: ',idx+1
			X_batch,Y_batch=generate_batch(X[i:i+16],Y[i:i+16],retargeted_width,height)
			loss=model.train_on_batch(X_batch,Y_batch)			
			print 'Loss: ',loss,' Accuracy: '
		model.save_weights('weights'+str(ep)+'.h5')
	return 

def resize(image):
    import tensorflow as tf 
    resized_image = tf.image.resize_images(image, (height,int(ar*width)))
    return resized_image

def repeat(col_vector,axis=0):
	return K.repeat_elements(col_vector,height,axis=1)
	
def merge(r1,r2):
	print r1.shape , r2.shape
	return tf.add(r1,r2)

def model_endtoend():
	retargeted_width=int(ar*width)
	img='/home/himanshu/task/VOCdevkit/VOC2007/JPEGImages/000020.jpg'
	im=np.array(Image.open(img,'r').resize((width,height)),dtype=np.float32)
	im=np.expand_dims(im,axis=0)
	model=encoder()
	model.add(decoder())
	model.add(Lambda(resize, input_shape=(height,width,depth), output_shape=(height,retargeted_width,depth),name='resize1'))
	model.add(conv1d_model(retargeted_width))
	model.add(Lambda(repeat,input_shape=(1,retargeted_width,depth),output_shape=(height,retargeted_width,depth),name='repeat1'))
	repeat1=model.layers[32].output
	resize1=model.layers[34].output
	model.add(Lambda(merge(resize1,repeat1),input_shape=(height,retargeted_width,depth),output_shape=(height,retargeted_width,depth),name='merge1'))

	print model.predict(im).shape

	
	model3=classification_model(int(ar*width),height)
	







	return 


if __name__=='__main__':
	retargeted_width = int(ar*width)
	# train_classification_model()
	model_endtoend()
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



