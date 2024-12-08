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
# DONE
entities = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 
            'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

num_img_train = 13
num_entity = len(entities)
num_img_total = num_img_train * num_entity

img_height = 116
img_width = 98
img_size = img_height * img_width

img_matrix = np.zeros((img_size, num_img_total)) # M x N

imgs = []
for img in os.listdir('./unpadded'): # use directory name directly
    if not(img.startswith('subject14') or img.startswith('subject15')): # only use 1st 13 for training
        imgs.append(img)
imgs.sort()
# for i in imgs:
#     print(i)

for idx, img in enumerate(imgs):
    face = io.imread(os.path.join('./unpadded', img))
    img_matrix[:, idx] = face.flatten()
# print(img_matrix[0])

# check if total # of images matches expected #
if img_matrix.shape == (img_size, num_img_total):
    print('Part 1: Pass - Data loaded successfully with correct dimensions.')
else:
    print('Part 1: Fail - Data dimensions or loading process may have issues.')


# 2 TODO: Generate matrix X
# DONE
mean_vector = np.zeros((img_size, 1))
for i in range(img_size):
    mean = np.sum(img_matrix[i]) / num_img_total # mean value of each pixel
    mean_vector[i] = mean

X = img_matrix - np.matmul(mean_vector, np.ones((1, num_img_total))) # mean-subtracted matrix

plt.imshow(mean_vector.reshape(img_height, img_width), cmap='gray')
plt.title('Mean Image')
plt.savefig('mean.png')
# plt.show()

# check if mean of columns of X is close to 0
if np.allclose(np.mean(X, axis=1), 0, atol=1e-6):
    print('Part 2: Pass - Data matrix X is correctly mean-centered.')
else:
    print('Part 2: Fail - Data matrix X is not mean-centered properly.')


# 3 TODO: Compute SVD of X & plot singular values
# IN PROGRESS
U, S, VT = np.linalg.svd(X)

singular_value_index = np.arange(num_img_total)
plt.figure() # new figure
plt.scatter(singular_value_index, S, label='Singular Value', s=3) # relatively small dot size for readability
plt.title('Singular Values of Matrix X')
plt.xlabel('Image Number')
plt.ylabel('Singular Value')
plt.legend()
plt.grid()
plt.savefig('singular_value.png')
# plt.show()

# check shapes of U, S, & V transpose
if np.shape(U) == (img_size, img_size) and np.shape(S) == (num_img_total,) and VT.shape == (num_img_total, num_img_total):
    print('Part 3: Pass - SVD computation completed with correct matrix dimensions.')
else:
    print('Part 3: Fail - SVD matrix dimensions do not match expectations.')

# check if cumulative variance sums to 1 (within numerical tolerance)
singular_values_squared = S ** 2
total_variance = np.sum(singular_values_squared)
cumulative_variance = np.cumsum(singular_values_squared) / total_variance
if np.isclose(cumulative_variance[-1], 1.0, atol=1e-6):
    print('Part 3: Pass - Cumulative variance is correctly computed.')
else:
    print('Part 3: Fail - Cumulative variance computation may have issues.')


# 4 TODO: Limit # of features & plot og image w/ aforementioned features




# 5 TODO: Test dataset




# 6 TODO: Repeat test w/ rotated image