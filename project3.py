import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io # using v3 in consideration of deprecation warning
import os

# Notes for ourselves:
# x_i = each image
# M = # of pixels i.e. length & width
# N = # of images (13 subjects for training * # of facial expressions & lighting conditions)
# X = image matrix (M x N)


# 1 TODO: Download dataset & load training dataset
# DONE (X vs img_matrix needs scrutiny)
entities = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 
            'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

num_img_train = 13
num_entity = len(entities)
num_img_total = num_img_train * num_entity

img_height = 116
img_width = 98
img_size = img_height * img_width

X = np.zeros((img_size, num_img_total)) # M x N

imgs = []
for img in os.listdir('./unpadded'):
    if not(img.startswith('subject14') or img.startswith('subject15')): # only use 1st 13 for training
        imgs.append(img)
imgs.sort()
# for i in imgs:
#     print(i)

for idx, img in enumerate(imgs):
    face = io.imread(os.path.join('./unpadded', img))
    X[:, idx] = face.flatten()
# print(X[0])

# check if total # of images matches expected #
if X.shape == (img_size, num_img_total):
    print('Part 1: Pass - Data loaded successfully with correct dimensions.')
else:
    print('Part 1: Fail - Data dimensions or loading process may have issues.')


# 2 TODO: Generate matrix X

# feature mean
# X


# 3 TODO: Compute SVD of X & plot singular values




# 4 TODO: Limit # of features & plot og image w/ aforementioned features




# 5 TODO: Test dataset




# 6 TODO: Repeat test w/ rotated image