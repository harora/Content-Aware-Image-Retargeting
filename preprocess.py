import os
import cv2
import PIL
import numpy as np
import xml.etree.ElementTree as ET 

nb_images=5008
n_classes=20

VOC_dataset='/home/himanshu/task/VOCdevkit/VOC2007/JPEGImages/'
xml_path='/home/himanshu/task/VOCdevkit/VOC2007/Annotations/'


classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}



def load_data(): 
    images=[]
    for f in os.listdir(dataset):
        images.append(dataset + f)
    return images

def generate_batch(X,Y,w,h):
    '''
    Returns batch of input images and labels
    Input: 
    X  : list of input images path in the batch
    Y  : list of labels of images in the batch
    w,h: dimensions of image
    '''
    
    X1=[]
    Y1=[]
    for img,label in zip(X,Y):
        im=np.array(PIL.Image.open(img,'r').resize((w,h)),dtype=np.float32)

        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        X1.append(im)
        Y1.append(label)

    X1=np.asarray(X1)
    Y1=np.asarray(Y1)
    
    return X1,Y1



def parse_xml(xml_file):
    '''
    extracts object id from xml file of each image
    '''
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
    '''
    generates and returns label for VOC2007 dataset
    '''
    X=[]
    Y=[]
    for f in os.listdir(VOC_dataset):
        X.append(VOC_dataset+f)
        label=parse_xml(xml_path+f.split('.')[0]+'.xml')
        lb=[0]*n_classes #Initializing label
        for l in label:
            if lb[l]==0:
                lb[l]=1
        Y.append(lb)

        if len(X)==nb_images: #VOC2007 nb_images
            break
    return X,Y





if __name__=='__main__':
    # load_VOC_data()
    # parse_xml()
