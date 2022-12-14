# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-


import cv2
"""
Dataset loading for ITK images and their respective labels within their metadata
header

"""
# Author: Nicolas Toussaint <nicolas.toussaint@gmail.com>
# 	  King's College London, UK

import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset



def _is_image_file(filename):
    """
    Is the given extension in the filename supported ?
    """
    # FIXME: Need to add all available SimpleITK types!
    IMG_EXTENSIONS = ['.nii.gz', '.nii', '.mha', '.mhd']
    IMG_EXTENSIONS = ['.jpg']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(fname):
    """
    Load supported image and return the loaded image.
    """
    return sitk.ReadImage(fname)


# +
from PIL import Image
import numpy as np

from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
def load_image(fname):
    ""
    image = Image.open(fname)
    image_size=(600,400)
    image = image.resize(image_size)
# import PIL
# Show in didferetn windo
# image.show()

# na = np.array(image, dtype=np.uint8)
# print(na.shape)
    na = np.array(image, dtype=np.uint8)    
    return na
    


# -

def save_image(itk_img, fname):
    """
    Save ITK image with the given filename
    """
    sitk.WriteImage(itk_img, fname)


def load_metadata(itk_img, key):
    """
    Load the metadata of the input itk image associated with key.
    """
    return itk_img.GetMetaData(key) if itk_img.HasMetaDataKey(key) else None


# +
import os
def extractlabelfromfile(fname):
    lbl_fname=fname.replace('.jpg','.txt')
    print('open ',lbl_fname)
#     with io.open("path/to/file.txt", mode="r", encoding="utf-8") as f
    file_size = os.path.getsize(lbl_fname)
    if (file_size > 0):
#         lbl_fname='//ethan/newshare/classwork/data2/train/train/1/553.txt'
        with open(lbl_fname, encoding="utf8") as f:
                fh=f.readlines()
                print('lines',len(fh))
        #         print(a)
                l_ret=[]
                for line in fh:
        #             print(line)
                    tokens=line.split()
#                     print(tokens)
                    k,x,y,w,h=tokens
#                     print(tokens)
                    l_ret.append(k)
                return l_ret
    else:
        print('no label')
    return None


def load_labels(fname):
    ""
    l=extractlabelfromfile(fname)
    return l


# -

def find_classes(filenames):
    classes = [extractlabelfromfile(f) for f in filenames]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def calculate_class_cardinality(filenames):
    _, class_to_idx = find_classes(filenames=filenames)
    classes = [extractlabelfromfile(f) for f in filenames]
    classes = [class_to_idx[l] for l in classes]
    _, counts = np.unique(classes, return_counts=True)
    return counts


def calculate_sample_weights(filenames):
    _, class_to_idx = find_classes(filenames=filenames)
    classes = [extractlabelfromfile(f) for f in filenames]
    classes = [class_to_idx[l] for l in classes]
    _, counts = np.unique(classes, return_counts=True)
    prob = counts / float(np.sum(counts))
    reciprocal_weights = [prob[classes[index]] for index in range(len(classes))]
    weights = (1. / np.array(reciprocal_weights))
    weights = weights / np.sum(weights)
    return weights


# +
def _is_image_file(filename):
    """
    Is the given extension in the filename supported ?
    """
    # FIXME: Need to add all available SimpleITK types!
#     IMG_EXTENSIONS = ['.nii.gz', '.nii', '.mha', '.mhd']
    IMG_EXTENSIONS = ['.jpg',]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def gglob(path, regexp=None):
    """Recursive glob
    """
    import fnmatch
    import os
    matches = []
    if regexp is None:
        regexp = '**'
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in fnmatch.filter(filenames, regexp):
            matches.append(os.path.join(root, filename))
    return matches

# Get filenames of all the available images
root='//ethan/newshare/classwork/data2/train/train'
# filenames = [y for y in gglob(root) if _is_image_file(y)]

# +
# # !cat //ethan/newshare/classwork/data2/train/train/1/476.txt
# -

# # !pwd



# +
l_ret=[]           
# # !cat //ethan/newshare/classwork/data2/train/train/1/553.txt
lbl_fname='//ethan/newshare/classwork/data2/train/train/1/553.txt'
with open(lbl_fname, encoding="utf8") as f:
        fh=f.readlines()
        print('lines',len(fh))
#         print(a)
        l_ret=[]
        for line in fh:
#             print(line)
            tokens=line.split()
            print(tokens)
            k,x,y,w,h=tokens
            print(tokens)
            l_ret.append(k)
#         return l_ret

k,x,y,w,h=tokens            



# open images
lbl_fname='//ethan/newshare/classwork/data2/train/train/1/553.jpg'
fname=lbl_fname.replace('.txt','.jpg')
from PIL import Image
import numpy as np

from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

image = Image.open(fname)
print(image)
# image = image.resize(image_size)
import PIL
# Show in didferetn windo
# image.show()

na = np.array(image, dtype=np.uint8)
print(na.shape)

# image_size=(800,500)
# image=image.resize((image_size))

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
imshow(image)


im_na=na

W,H,CHAN=na.shape
print(W,H,CHAN)

print(k,x,y,w,h)
print(np.floor(float(x)*W))

# +
import numpy as np
import cv2
import os
im=im_na
im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

# # therasholding
# ret,thresh = cv2.threshold(im2,127,255,0)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# print('num of contours',len(contours))
# for i in range(0, len(contours)):
#     if (i % 2 == 0):
#        cnt = contours[i]
#        #mask = np.zeros(im2.shape,np.uint8)
#        #cv2.drawContours(mask,[cnt],0,255,-1)
#        cnt_x,cnt_y,cnt_w,cnt_h = cv2.boundingRect(cnt)
        
#        

xx=int(np.floor(float(x)*W))
yy=int(np.floor(float(y)*H))
ww=int(np.floor(float(w)*W))
hh=int(np.floor(float(h)*H))

# print('scaling',x,xx,W)
print('scaling')
# print(np.floor(float(x),W))
print(float(x),x)


# yy,xx,ww,hh = np.floor(x*W),np.floor(y*H),np.floor(w*W),np.floor(h*H)
# print(yy)

ROI = im[xx-ww:xx+ww,yy-hh:yy+hh]

mask_size=im.shape
mask = np.zeros(mask_size, dtype=np.uint8)
# mask = 0
mask[xx-ww:xx+ww,yy-hh:yy+hh]=1

# print(xx,yy,ww,hh)
# im.shape
print('ROI',ROI.shape,xx,yy,ww,hh)

print(xx,yy,ww,hh)

# if (1):
#     cv2.imshow('Features', ROI)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows() 

name=os.path.basename(fname)+'.roi.jpg'
print('saving to ',name)
# cv2.imwrite(name,ROI)
imshow(mask)

# +
ROI.shape

mask_size = image_size
mask = np.array(mask, dtype=np.uint8)
image_size

# +
im=ROI
# therasholding


im[im == 255] = 1
im[im == 0] = 255
im[im == 1] = 0
# im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(im2,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print('num of contours',len(contours))
for i in range(0, len(contours)):
        #     if (i % 2 == 0):
        cnt = contours[i]
        mask = np.zeros(im2.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
#         print(cnt.shape)
        cnt_x,cnt_y,cnt_w,cnt_h = cv2.boundingRect(cnt)
#         print(cnt_x,cnt_y,cnt_w,cnt_h)
        
        
# imshow(im2)
# imshow(im2)
im.shape
# im2.shape
# -

# # !ls *.jpg
# im=ROI
# im.shape,im[0:200]




# +
# filenames = [y for y in gglob(root) if _is_image_file(y)]
# filenames

# +
# dataset=FetalUltrasoundDataset('//ethan/newshare/classwork/data2/train')

# # \\ethan\newshare\classwork\data2\train\train

# +
# img,lbl=dataset[0]
# # img
# # print(lbl)
# # imshow(img)
# imshow(img)
# # lbl

# +
IMG_EXTENSIONS = ['.jpg']
class FetalUltrasoundDataset(Dataset):
    """
    Arguments
    ---------
    root : string
        Root directory of dataset. The folder should contain all images for each
        mode of the dataset ('train', 'validate', or 'infer'). Each mode-version
        of the dataset should be in a subfolder of the root directory

        The images can be in any ITK readable format (e.g. .mha/.mhd)
        For the 'train' and 'validate' modes, each image should contain a metadata
        key 'Label' in its dictionary/header

    mode : string, (Default: 'train')
        'train', 'validate', or 'infer'
        Loads data from these folders.
        train and validate folders both must contain subfolders images and labels while
        infer folder needs just images subfolder.
    transform : callable, optional
        A function/transform that takes in input itk image or Tensor and returns a
        transformed
        version. E.g, ``transforms.RandomCrop``

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        # training set or test set

        assert(mode in ['train', 'validate', 'infer'])

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(root, 'train')
        elif mode == 'validate':
            self.root = os.path.join(root, 'validate')
        else:
            self.root = os.path.join(root, 'infer') if os.path.exists(os.path.join(root, 'infer')) else root

        def gglob(path, regexp=None):
            """Recursive glob
            """
            import fnmatch
            import os
            matches = []
            if regexp is None:
                regexp = '**'
            for root, dirnames, filenames in os.walk(path, followlinks=True):
                for filename in fnmatch.filter(filenames, regexp):
                    matches.append(os.path.join(root, filename))
            return matches

        # Get filenames of all the available images
        self.filenames = [y for y in gglob(self.root) if _is_image_file(y)]
#         self.filenames = [y for y in gglob(self.root, '*.*') if _is_image_file(y)]

        if len(self.filenames) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.filenames.sort()

        self.transform = transform
        self.target_transform = target_transform
        
        # find_classes
#         classes, class_to_idx = find_classes(self.filenames)
#         self.labels

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.class_cardinality = calculate_class_cardinality(self.filenames)
#         self.sample_weights = calculate_sample_weights(self.filenames)
      
    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (image, label) where label the organ apparent in the image
        """

        image = load_image(self.filenames[index])
#         labels = None
        labels = load_labels(self.filenames[index])

# #         label = load_metadata(image, 'Label')
#         label = load_label(self.filenames[index])
#         if label is not None:
#             labels = [0] * len(self.classes)
#             labels[self.class_to_idx[label]] = 1
#             labels = np.array(labels, dtype=np.float32)

#         if self.transform is not None:
#             image = self.transform(image)
#         if self.target_transform is not None:
#             labels = self.target_transform(labels)

        if (self.mode == 'infer') or (labels is None):
            return image, []
        else:
            return image, labels

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_root(self):
        return self.root

    def get_classes(self):
        return self.classes

    def get_class_cardinality(self):
        return self.class_cardinality
    
    def get_sample_weights(self):
        return self.sample_weights
# -






