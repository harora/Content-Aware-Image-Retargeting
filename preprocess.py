from PIL import Image
from os import listdir
import numpy as np
import xml.etree.ElementTree as ET 
import cv2


dataset='/home/himanshu/code/color/ILSVRC2015/Data/DET/test/'
VOC_dataset='/home/himanshu/task/VOCdevkit/VOC2007/JPEGImages/'
xml_path='/home/himanshu/task/VOCdevkit/VOC2007/Annotations/'

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}

n_classes=20
def load_data():
	images=[]
	for f in listdir(dataset):
		images.append(dataset+f)
		if len(images) == 3:
			break
		print f
	return images

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_suf=''
    image_path=''
    labels=[]

    for item in root:
        if item.tag=='object':
            obj_name = item[0].text
            obj_num = classes_num[obj_name]
            labels.append(obj_num)

    return labels

def load_VOC_data():
    X=[]
    Y=[]
    for f in listdir(VOC_dataset):
        X.append(VOC_dataset+f)
        label=parse_xml(xml_path+f.split('.')[0]+'.xml')
        lb=[0]*n_classes
        for l in label:
            if lb[l]==0:
                lb[l]=1
        Y.append(lb)

        if len(X)==5008:
            break
    return X,Y

def generate_batch(X,Y,w,h):
    X1=[]
    Y1=[]
    for img,label in zip(X,Y):
        im=np.array(Image.open(img,'r').resize((w,h)),dtype=np.float32)

        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        X1.append(im)
        Y1.append(label)

    X1=np.asarray(X1)
    Y1=np.asarray(Y1)
    return X1,Y1



if __name__=='__main__':
	load_VOC_data()
	# parse_xml()
