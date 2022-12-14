# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Siamese Segmentation Training

# ## Imports

# Jupyter Notebook utils
# %load_ext autoreload
# %matplotlib inline

# +
import time
import os

from tqdm.notebook import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt

import src.img_utils as utils
import src.model as siam_models
# from src.datasets import iCoSegDataset
# %autoreload 2
# -

# ## Dataset Loading

# ### Constants

# +
# datap='c:/Users/yshao/Desktop/projects-2020/Worsning/Imaging_object/CMU_Cornell_iCoseg_dataset/dataset_public'

# +
# datap=''

BATCH_SIZE = 2 # TODO: Allow changing
NUM_WORKERS = 1

# IMAGES_PATH = f"{datap}/images_subset_test"
# MASKS_PATH = f"{datap}/ground_truth_subset_test"

VALIDATION_SPLIT = 0.2 # What % the validation set should be
SHUFFLE = True
# -



# ### Instantiation





# +
##
## Tumor Dataset Loader
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import SubsetRandomSampler

import numpy as np
# %matplotlib inline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Computation Details')
print(f'\tDevice Used: ({device})  {torch.cuda.get_device_name(torch.cuda.current_device())}\n')

print('Packages Used Versions:-')
print(f'\tPytorch Version: {torch.__version__}')

# To Start TensorBoard
# tensorboard --logdir logs --samples_per_plugin images=200

# -







# +
import sys
hostp='/workspace/ultrasound/newrun'
sys.path.append(hostp)
# from fetalnav import fetalnav
# from fetalnav import fetalnav
# from fetalnav.transforms import itk_transforms as itktransforms
# from fetalnav.transforms import tensor_transforms as tensortransforms
# from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification
from fetalnav.datasets.dataset import FetalUltrasoundDataset


# from fetalnav.models.resnet import resnet18


# +
# Database
import glob
filep='//ethan/newshare/classwork/data2/train/**/[2-4]/*.jpg'

filep='//ethan/newshare/classwork/data2/train/sample/*.jpg'

filep='//ethan/newshare/classwork/data2/train/sample_mix/*.jpg'

filep='/workspace/ultrasound/sample/*.jpg'

filep='/workspace/ultrasound/test_all/*.jpg'

lfiles=glob.glob(filep)
# dataset=FetalUltrasoundDataset(filep,image_size=(288,188))
dataset=FetalUltrasoundDataset(filep)
print('Training lengh',len(lfiles))

# +
filep='/workspace/ultrasound/test_all/*.jpg'

filep='/workspace/ultrasound/test/*.jpg'
# filep='//ethan/newshare/classwork/data2/train/sample_mix/*.jpg'
lfiles=glob.glob(filep)

print('est lengh',len(lfiles))

new_test=FetalUltrasoundDataset(filep)
# -





len(dataset),len(new_test)

VALIDATION_SPLIT=0.2

# +
import torch

NUM_WORKERS=1

# Test/train dataset split
# Default is 80% training, 20% testing
dataset_length = len(dataset)
train_size = int((1.0 - VALIDATION_SPLIT) * dataset_length)
test_size = dataset_length - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# +
print('datasizes')
print(train_size,test_size,dataset_length)

# train_dataset=train_dataset[0:3000]
# test_dataset=test_dataset[0:200]

# len(train_dataset),train_dataset.__class__
# len(dataset),dataset.__class__
SHUFFLE=True
# -

# Loader
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)

from tqdm.notebook import tqdm
# next(iter(loader))
torch.cuda.empty_cache()
loader = tqdm(enumerate(train_loader), desc="Batch Progress: ", total=int((train_size/BATCH_SIZE)))

for i, sample in loader:
    print(i)
    continue

newloader = DataLoader(
    new_test,
#     batch_size=BATCH_SIZE,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)



# +
# check one data item

item=dataset[0]

# -

if (0):
    ee=enumerate(train_loader)
    for e in list(ee):
        print(e[0],e[1].keys())
        item=e[1]
        print(e[0],item['image'].shape,item['label'],len(item['mask']))

# +
# for e in (list(ee)):
#     print(e
# list(ee)
# -



# ## Training

# ### Constants

# +
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0004

SAVE_WEIGHTS = True
WEIGHTS_PATH = "./weights"

MARGIN_ALPHA = 0.25 # Parameter used in loss calculation

EPOCHS = 100
# -

# ### Preparation

# +
# Different Model
tag=''

prev_loss = float("inf")

# VGG16
model = siam_models.CoSegNet();
tag=tag+'.vgg16'

# ResNet
# model = siam_models.CoSegResNet();
# tag=tag+'.resnet'

# UNet
# model = siam_models.CoSegUNet();
# tag=tag+'.unet'

MARGIN_ALPHA
model

# +


# Instantiation
device = None
if torch.cuda.is_available():
    print("Using CUDA 0 as device")
    device = torch.device("cuda:1")
else:
    print("Using CPU as device")
    device = torch.device("cpu")

prev_loss = float("inf")
# model = siam_models.CoSegNet();

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)
model.to(device);

# Citerions
criterion_bce = nn.BCELoss()



# criterion_triplet = nn.TripletMarginLoss(margin=MARGIN_ALPHA)
# Using Cosine Embedding Loss as a similar loss to triplet
# due to prescence of pairs
criterion_cel = nn.CosineEmbeddingLoss(margin=MARGIN_ALPHA)
tag=tag+'.embedding'

# criterion_cel = nn.MarginRankingLoss(margin=MARGIN_ALPHA)
# tag=tag+'.margin'

# +
# # !ls -l weights/*.path*
# print(WEIGHTS_PATH)
# pp=!ls -1t  weights/*.path* | tail -1
# pp
# # !mkdir weights
# -

# Load Weight
if (0):
    try:
        modelp='UlTraSegNet_VGG16.path.trial=2021031911'
        modelp=pp[0]
#         model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, modelp)))
        model.load_state_dict(torch.load(modelp))
    except Exception as e:
        print(e)

# ### Training Loop

# +
# Look at one image

# sample=next(iter(loader))
# -



# +
# Test iteration here

imageA = sample["image"][0].unsqueeze(0)
imageB = sample["image"][1].unsqueeze(0)
# Mask Tensors
maskA = sample["mask"][0][0].float().unsqueeze(0).to(device)
maskA[maskA==255]=1
maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
maskB[maskB==255]=1
# Labels
labelA = sample["label"][0][0][0]
labelB = sample["label"][0][0][1]


#             picA = p(F.to_pil_image(sample['image'][0]))
#             picB = p(sample['image'][1])

norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()
#             
#             Normalizing images before sending to the model
#             Forming into batch-shape for processing
#             norm_imageA = F.normalize(imageA.repeat(3, 1, 1),
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)
#             norm_imageB = F.normalize(imageB.repeat(3, 1, 1), 
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)

#             norm_imageA = imageA
#             norm_imageB = imageB

norm_imageA = norm_imageA.to(device)
norm_imageB = norm_imageB.to(device)

# Network Run
pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)

# Obtaining co-segmentation masks
# based off of the decision by the decision net
# During testing, the decision net's value
# would be thresheld before multiplying
pred_maskA = pmapA*decision
pred_maskB = pmapB*decision            
# -

# check image size with model input size
norm_imageA.shape,pred_maskA.shape,maskA.shape

# +
# pred_maskA.shape,maskA.unsqueeze(0).reshape((1,1,384,576)).shape
from PIL import Image
import numpy as np

# PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

PIL_image = Image.fromarray(maskA.cpu().numpy().astype('uint8'), 'RGB')
# plt.imshow(PIL_image)
PIL_image
maskA

# +
# Network Run
pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)

# Obtaining co-segmentation masks
# based off of the decision by the decision net
# During testing, the decision net's value
# would be thresheld before multiplying
pred_maskA = pmapA*decision
pred_maskB = pmapB*decision

# Loss Calculations/Evaluations
# Configuring loss weights depending on sample
# Also deciding if we produce a groundtruth mask
truth = None
pairwise = None
if labelA == labelB: # Positive Sample found
    # Weighting loss evenly
    w1=w2=w3 = 0.33

    truth = 1
    pairwise = 1
else: # Negative sample
    # Prevent Loss1 from backpropogating
    w1 = 0
    w2=w3 = 0.5

    # Create a null mask from the groundtruths
    maskA = maskA * 0
    maskB = maskB * 0

    truth = 0
    pairwise = -1

# Loss 1
# Pixel-wise Binary Cross Entropy Loss
loss1A = criterion_bce(pred_maskA, maskA.unsqueeze(0))
loss1B = criterion_bce(pred_maskB, maskB.unsqueeze(0))

# Loss 2
# Standard Triplet Loss with Margin
pairwise = torch.tensor(pairwise).to(device)
loss2 = criterion_cel(vectorA.unsqueeze(0), vectorB.unsqueeze(0), pairwise)

# Loss 3
# Binary Cross Entropy Loss
truth = torch.tensor(truth).float().unsqueeze(0).to(device)
loss3 = criterion_bce(decision, truth)

loss_final = w1*(loss1A + loss1B) + w2*loss2 + w3*loss3

# +
# # loss1A,loss1B,loss2,loss3
# maskA[maskA==255]=1
# maskA.max()

loss1A,loss1B
# -



train_size/BATCH_SIZE,train_size,len(dataset)



# +
import os
try: os.makedirs('./logs')
except: ""

# summary
import datetime
# from torch.utils.tensorboard import SummaryWriter
t1 = datetime.datetime.now().strftime('%Y%m%d%H')
t1
# -

t1

losses=[]

# +

# writer = SummaryWriter('./logs/{}_{}_{}'.format(now.hour, now.minute, now.second))


for epoch in tqdm(range(EPOCHS), desc="Epoch Progress: "):
    # Losses and Loss Weights
    # Loss_final = W1*L1 + W2*L2 + W3*L3
    total_loss = 0
    loss_final = 0
    loss1A = 0 # Pixel-wise binary cross entropy
    loss1B = 0
    weight1 = 0
    loss2 = 0 # Triplet loss
    weight2 = 0
    loss3 = 0 # Cross Entropy
    weight3 = 0
    
    #Statistics
    predictions_correct = 0
    predictions_total = 0
    background_percent_correct = 0
    
    loader = tqdm(enumerate(train_loader), desc="Batch Progress: ", total=(train_size/BATCH_SIZE))
    
    model.train()
    
    time_start = time.time()
    
    for i, sample in loader:
        try:
            # Image Tensors
            imageA = sample["image"][0].unsqueeze(0)
            imageB = sample["image"][1].unsqueeze(0)
            # Mask Tensors
            maskA = sample["mask"][0][0].float().unsqueeze(0).to(device)
            maskA[maskA==255]=1
            maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
            maskB[maskB==255]=1
            # Labels
            labelA = sample["label"][0][0][0]
            labelB = sample["label"][0][0][1]

            norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
            norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


            norm_imageA = norm_imageA.to(device)
            norm_imageB = norm_imageB.to(device)

            # Network Run
            pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)

            # Obtaining co-segmentation masks
            # based off of the decision by the decision net
            # During testing, the decision net's value
            # would be thresheld before multiplying
            pred_maskA = pmapA*decision
            pred_maskB = pmapB*decision

            # Loss Calculations/Evaluations
            # Configuring loss weights depending on sample
            # Also deciding if we produce a groundtruth mask
            truth = None
            pairwise = None
            if labelA == labelB: # Positive Sample found
                # Weighting loss evenly
                w1=w2=w3 = 0.33

                truth = 1
                pairwise = 1
            else: # Negative sample
                # Prevent Loss1 from backpropogating
                w1 = 0
                w2=w3 = 0.5

                # Create a null mask from the groundtruths
                maskA = maskA * 0
                maskB = maskB * 0

                truth = 0
                pairwise = -1

            # Loss 1
            # Pixel-wise Binary Cross Entropy Loss
            loss1A = criterion_bce(pred_maskA, maskA.unsqueeze(0))
            loss1B = criterion_bce(pred_maskB, maskB.unsqueeze(0))

            # Loss 2
            # Standard Triplet Loss with Margin
            pairwise = torch.tensor(pairwise).to(device)
            loss2 = criterion_cel(vectorA.unsqueeze(0), vectorB.unsqueeze(0), pairwise)

            # Loss 3
            # Binary Cross Entropy Loss
            truth = torch.tensor(truth).float().unsqueeze(0).to(device)
            loss3 = criterion_bce(decision, truth)

            loss_final = w1*(loss1A + loss1B) + w2*loss2 + w3*loss3

            loss_final.backward()
            optimizer.step()

            total_loss = total_loss + loss_final.item()
        except Exception as e:
            print(e)
    
    print("Total Loss: " + str(total_loss))
    losses.append(total_loss)
    
    # Validation
#     valloader = tqdm(enumerate(test_loader), desc="Batch Progress: ", total=(train_size/BATCH_SIZE))
    
    
    if total_loss < prev_loss and SAVE_WEIGHTS:
        # Check for dir, create if it doesn't exist
        if not os.path.exists(WEIGHTS_PATH):
            os.makedirs(WEIGHTS_PATH)
        
        
        prev_loss = total_loss
        print("Saving Model")
#         torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, f"BrainSegNet_VGG16.path.loss={str(total_loss)[:6]}.trial={t1}"))
        torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, f"UlTraSegNet_VGG16.path.trial={t1}"))
        
    # Per-Epoch Update
#     if epoch >=2:
#         writer.add_scalar('training_loss',total_loss/len(dataset), epoch)    
        
        
    time_total = round(time.time() - time_start, 2)
    print('Total Time: ',time_total)
    print('Epoch #:',epoch)
# -

newloader = DataLoader(
    new_test,
#     batch_size=BATCH_SIZE,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)


# +
def validate():
    valloader = tqdm(enumerate(newloader), desc="Batch Progress: ", total=(train_size/BATCH_SIZE))
    model.eval()
    for i, sample in valloader:
        print(i)
        try:
            
            # Image Tensors
            imageA = sample["image"][0].unsqueeze(0)
            imageB = sample["image"][1].unsqueeze(0)
            # Mask Tensors
            maskA = sample["mask"][0][0].float().unsqueeze(0).to(device)
            maskA[maskA==255]=1
            maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
            maskB[maskB==255]=1
            # Labels
            labelA = sample["label"][0][0][0]
            labelB = sample["label"][0][0][1]

            norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
            norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


            norm_imageA = norm_imageA.to(device)
            norm_imageB = norm_imageB.to(device)

            # Network Run
            pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)
        except:
            ""
            continue


#             pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)


            # labelA.item(),labelB.item()

        diceA=get_dice_score(pmapA,maskA)
        diceB=get_dice_score(pmapB,maskB)
#             print(diceA,diceB)
        val.append(diceA)
        val.append(diceB)
    return val



# val_val=validate()
# val_val


# +
# Check model presence
output=f"UlTraSegNet_VGG16.path.trial={t1}"
print('saved',output)
# !ls {WEIGHTS_PATH}/{output}

# Loa from file
# if (0):
    

# +
####LOSSES
import pickle

with open(f"losses.{t1}.pkl",'wb') as f:
     pickle.dump(np.array(losses), f)

np_losses=np.array(losses)

print(f"fig1_training.png.{t1}")

# +
EPOCHS
# acc = history.history['accuracy']

# loss=history.history['loss']

epochs_range = range(EPOCHS)
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training Accuracy')

# plt.subplot(1, 1, 1)
ax.plot(epochs_range, losses, label='Training Loss')
ax.legend(loc='upper right')
ax.set_title('Training Loss (30Dataset)')
ax.set_xlabel('EPOCHS')
ax.set_ylabel('Training loss')
fig.show()
fig.savefig(f"fig1_training.{t1}.t30.png", bbox_inches='tight')
# -

# ## Testing

# +
import numpy as np
def _dice_coefficient(predicted, target):
    """Calculates the Sørensen–Dice Coefficient for a
    single sample.
    Parameters:
        predicted(numpy.ndarray): Predicted single output of the network.
                                Shape - (Channel,Height,Width)
        target(numpy.ndarray): Actual required single output for the network
                                Shape - (Channel,Height,Width)

    Returns:
        coefficient(float): Dice coefficient for the input sample.
                                    1 represents high similarity and
                                    0 represents low similarity.
    """
    smooth = 1
    product = np.multiply(predicted, target)
    intersection = np.sum(product)
    coefficient = (2*intersection + smooth) / \
        (np.sum(predicted) + np.sum(target) + smooth)
    return coefficient

import cv2

def get_dice_score(pmapB,maskB):
    pmapB_np=pmapB.cpu().detach().numpy().squeeze(0)
    maskB_np=maskB.cpu().detach().numpy()

    ret, bw_img = cv2.threshold(pmapB_np,0.06,1,cv2.THRESH_BINARY)

    dice_score=_dice_coefficient(maskB_np[0,:],bw_img[0,:])
    print('Dice',dice_score)
    return dice_score

# pmapB_np
# bw_img.shape,maskB_np.shape


# +

# len(test_loader)
# -

sample = next(iter(test_loader))
print(sample['label'])
sample = next(iter(test_loader))
print(sample['label'])

# +


# # Test Instance
sample = next(iter(test_loader))

sample

# +

# Image Tensors
imageA = sample["image"][0].unsqueeze(0)
imageB = sample["image"][1].unsqueeze(0)
# Mask Tensors
maskA = sample["mask"][0][0].float().unsqueeze(0).to(device)
maskA[maskA==255]=1
maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
maskB[maskB==255]=1
# Labels
labelA = sample["label"][0][0][0]
labelB = sample["label"][0][0][1]


norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


norm_imageA = norm_imageA.to(device)
norm_imageB = norm_imageB.to(device)

# +
model.eval()
pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA, norm_imageB)


# labelA.item(),labelB.item()

get_dice_score(pmapA,maskA),get_dice_score(pmapB,maskB)

# +
# loader = tqdm(enumerate(newloader), desc="Batch Progress: ", total=(len(newloader)/BATCH_SIZE))

# len(newloader)

# +
### Get newgative

filep='/workspace/ultrasound/test_negative/*.jpg'
# filep='//ethan/newshare/classwork/data2/train/sample_mix/*.jpg'
lfiles=glob.glob(filep)

print('est lengh',len(lfiles))

new_test_neg=FetalUltrasoundDataset(filep)


newloader_neg = DataLoader(
    new_test_neg,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)

# # Ground Instance
sample = next(iter(newloader_neg))

# Image Tensors
imageA = sample["image"][0].unsqueeze(0)
norm_imageA_neg=imageA.repeat(3, 1, 1).unsqueeze(0).float()
norm_imageA_neg=norm_imageA_neg.to(device)

# +
### Build New Loader landmarks ###

### Get newgative

filep='/workspace/ultrasound/landmark_img/*.jpg'
# filep='//ethan/newshare/classwork/data2/train/sample_mix/*.jpg'
lfiles=glob.glob(filep)

print('est lengh',len(lfiles))

new_test_land=FetalUltrasoundDataset(filep)

BATCH_SIZE=1
newloader_land = DataLoader(
    new_test_land,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)

# sample = next(iter(newloader_neg))

# # # Ground Instance
# sample = next(iter(newloader_neg))

# # Image Tensors
# imageA = sample["image"][0].unsqueeze(0)
# norm_imageA_neg=imageA.repeat(3, 1, 1).unsqueeze(0).float()
# norm_imageA_neg=norm_imageA_neg.to(device)

loader_test = tqdm(enumerate(newloader_land), desc="Batch Progress: ", total=(len(newloader_land)/BATCH_SIZE))

len(newloader_land)

# +
# # !ls ../../
# len(loader_test)

# +
l_score=[]
cls=0
for i, sample in loader_test:
    print('i',i)
    ## TestBench
 
    imageB = sample["image"][0].unsqueeze(0)
    # imageB = sample["image"][1].unsqueeze(0)
    # Mask Tensors
    maskB = sample["mask"][0][0].float().unsqueeze(0).to(device)
    maskB[maskB==255]=1
    # maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
    # maskB[maskB==255]=1
    # Labels
    labelB = sample["label"][0][0][0]
    # labelB = sample["label"][0][0][1]


    norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
    norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


    norm_imageA = norm_imageA.to(device)
    norm_imageB = norm_imageB.to(device)


    model.eval()
    pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA_neg, norm_imageB)

#     l_score.append(get_dice_score(pmapA,maskA))
    dice=get_dice_score(pmapB,maskB)    
    if (dice < 0.05):
        cls=cls+1
    l_score.append(get_dice_score(pmapB,maskB))
# -

# positive Test
# np.mean(np.array(l_score))
cls,len(l_score),'mean DICE',np.mean(l_score) # ('mean DICE', 0.5450576740829148) - 15
# (80, 'mean DICE', 0.5454188200481974)
(5, 79, 'mean DICE', 0.5458560788438688)

# +
# positive Neg
# (80, 'mean DICE', 0.5454188200481974)
# # !ls /workspace/ultrasound/test_all_neg

# +
### Build New Loader landmarks ###

### Get newgative

filep='/workspace/ultrasound/test_negative/*.jpg'
# filep='//ethan/newshare/classwork/data2/train/sample_mix/*.jpg'
lfiles=glob.glob(filep)

print('est lengh',len(lfiles))

new_test_neg=FetalUltrasoundDataset(filep)

BATCH_SIZE=1
newloader_neg = DataLoader(
    new_test_neg,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE,
    drop_last=True
)

loader_test_neg = tqdm(enumerate(newloader_neg), desc="Batch Progress: ", total=(len(newloader_neg)/BATCH_SIZE))

# len(newloader_land)

# +
l_score=[]
for i, sample in loader_test_neg:
    print('i',i)

    imageB = sample["image"][0].unsqueeze(0)
    # imageB = sample["image"][1].unsqueeze(0)
    # Mask Tensors
    maskB = sample["mask"][0][0].float().unsqueeze(0).to(device)
    maskB[maskB==255]=1
    # maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
    # maskB[maskB==255]=1
    # Labels
    labelB = sample["label"][0][0][0]
    # labelB = sample["label"][0][0][1]


    norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
    norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


    norm_imageA = norm_imageA.to(device)
    norm_imageB = norm_imageB.to(device)


    model.eval()
    pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA_neg, norm_imageB)

#     l_score.append(get_dice_score(pmapA,maskA))
    l_score.append(get_dice_score(pmapB,maskB))

# +
# negatives
# 7, 7

# +
# # Test-set


sample = next(iter(newloader_land))
# Image Tensors
imageB = sample["image"][0].unsqueeze(0)
# imageB = sample["image"][1].unsqueeze(0)
# Mask Tensors
maskB = sample["mask"][0][0].float().unsqueeze(0).to(device)
maskB[maskB==255]=1
# maskB = sample["mask"][0][1].float().unsqueeze(0).to(device)
# maskB[maskB==255]=1
# Labels
labelB = sample["label"][0][0][0]
# labelB = sample["label"][0][0][1]


norm_imageA=imageA.repeat(3, 1, 1).unsqueeze(0).float()
norm_imageB=imageB.repeat(3, 1, 1).unsqueeze(0).float()


norm_imageA = norm_imageA.to(device)
norm_imageB = norm_imageB.to(device)


model.eval()
pmapA, pmapB, vectorA, vectorB, decision = model(norm_imageA_neg, norm_imageB)

# +
# # mapA = F.to_pil_image(pmapA.detach().cpu())

# imageA_cpu = F.to_pil_image(imageA.cpu().squeeze().unsqueeze(0))
# plt.imshow(imageA_cpu)

# +
print('Test Image')

imageB_cpu = F.to_pil_image(imageB.cpu().squeeze().unsqueeze(0))
plt.imshow(imageB_cpu)

# +
# print('Ground Image A')

# mapA = F.to_pil_image(pmapA.detach().cpu().squeeze().unsqueeze(0))

# imageA_cpu = F.to_pil_image(imageA.cpu().squeeze().unsqueeze(0))
# plt.imshow(imageA_cpu)
# plt.imshow(mapA, cmap="jet", alpha=0.4)
# plt.plot()

# print(decision)


# +
# # Mask
# maskA_cpu = F.to_pil_image(maskA.cpu().squeeze().unsqueeze(0))

# # imageA_cpu = F.to_pil_image(imageA.cpu().squeeze())
# plt.imshow(imageA_cpu)
# plt.imshow(maskA_cpu, cmap="jet", alpha=0.4)
# plt.plot()

# print(decision)

# +


mapB = F.to_pil_image(pmapB.detach().cpu().squeeze().unsqueeze(0))

imageB_cpu = F.to_pil_image(imageB.cpu().squeeze().unsqueeze(0))
plt.imshow(imageB_cpu)
plt.imshow(mapB, cmap="jet", alpha=0.4)
plt.plot()

get_dice_score(pmapB,maskB)
# -



# +
# Ground truth Mask
maskB_cpu = F.to_pil_image(maskB.cpu().squeeze().unsqueeze(0))

# imageA_cpu = F.to_pil_image(imageA.cpu().squeeze())
plt.imshow(imageB_cpu)
plt.imshow(maskB_cpu, cmap="jet", alpha=0.4)
plt.plot()

print(decision)

# -





# +
# # plt.imshow(F.to_pil_image(bw_img))
# bw_img.shape,maskB_np.shape
# plt.imshow(bw_img[0,:], cmap="gray")
# # bw_img[0,:].shape

# +
# plt.imshow(maskB_np[0,:], cmap="gray")

# +
# # plt.imshow(mapB)
# pmapB_np.max()
# pmapB_np.min()
# pmapB_np.mean()
# -


