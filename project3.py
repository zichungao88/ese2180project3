import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io # using v3 in consideration of deprecation warning
import os

# Notes for ourselves:
# x_i = each image
# M = # of pixels i.e. length * width
# N = # of images (13 subjects for training * # of facial expressions & lighting conditions)
# X = image matrix (M x N)


# 1 TODO: Download dataset & load training dataset
# DONE
entities = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 
            'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

num_subject_train = 13
num_entity = len(entities)
num_img_train = num_subject_train * num_entity

img_height = 116
img_width = 98
img_size = img_height * img_width

img_train_matrix = np.zeros((img_size, num_img_train)) # M x N

imgs_train = []
for i in os.listdir('./unpadded'): # use directory name directly
    if not(i.startswith('subject14') or i.startswith('subject15')): # only use 1st 13 for training
        imgs_train.append(i)
imgs_train.sort()
# for i in imgs_train:
#     print(i)

for i, j in enumerate(imgs_train):
    face = io.imread(os.path.join('./unpadded', j))
    img_train_matrix[:, i] = face.flatten()
# print(img_train_matrix[0])

# check if total # of images matches expected #
if img_train_matrix.shape == (img_size, num_img_train):
    print('Part 1: Pass - Data loaded successfully with correct dimensions.\n')
else:
    print('Part 1: Fail - Data dimensions or loading process may have issues.\n')


# 2 TODO: Generate matrix X
# DONE
mean_vector = np.zeros((img_size, 1))
for i in range(img_size):
    mean = np.sum(img_train_matrix[i]) / num_img_train # mean value of each pixel
    mean_vector[i] = mean

X = img_train_matrix - mean_vector @ np.ones((1, num_img_train)) # mean-subtracted matrix

plt.imshow(mean_vector.reshape(img_height, img_width), cmap='gray')
plt.title('Mean Image')
plt.savefig('mean.png')
# plt.show()

# check if mean of columns of X is close to 0
if np.allclose(np.mean(X, axis=1), 0, atol=1e-6):
    print('Part 2: Pass - Data matrix X is correctly mean-centered.\n')
else:
    print('Part 2: Fail - Data matrix X is not mean-centered properly.\n')


# 3 TODO: Compute SVD of X & plot singular values
# DONE
U, S, VT = np.linalg.svd(X)

singular_value_index = np.arange(num_img_train)
plt.figure() # new figure
plt.scatter(singular_value_index, S, label='Singular Value', s=3) # relatively small dot size for readability
plt.title('Singular Values of Matrix X')
plt.xlabel('Image Number')
plt.ylabel('Singular Value')
plt.legend()
plt.grid()
plt.savefig('singular_value.png')
# plt.show()

# # of principal components = r = # of nonzero singular values
# must implement threshold to determine which values to not consider as singular values
# since all diagonal entries of Matrix Sigma are nonzero but some are very very close to 0
principal_components_train = []
num_principal_component = 0
for i in S:
    if i >= 0.001: # self-defined hard-coded cutoff/threshold value
        principal_components_train.append(i)
        num_principal_component += 1
principal_components_train = np.array(principal_components_train)
# print(num_principal_component)
# print(principal_components)

# check how many principal components are needed to capture a certain amount of training image data
# no need to recreate array w/ only principal components since singular values of 0 do not contribute to total variance
singular_values_squared = S ** 2
total_variance = np.sum(singular_values_squared)
cumulative_variance = np.cumsum(singular_values_squared) / total_variance

capture70 = 0
capture80 = 0
capture90 = 0
capture95 = 0
for i in range(len(cumulative_variance)):
    if cumulative_variance[i] >= 0.95 and capture95 == 0:
        capture95 = i + 1
    elif cumulative_variance[i] >= 0.9 and capture90 == 0:
        capture90 = i + 1
    elif cumulative_variance[i] >= 0.8 and capture80 == 0:
        capture80 = i + 1
    elif cumulative_variance[i] >= 0.7 and capture70 == 0:
        capture70 = i + 1

print(str(capture70) + ' principal components are needed to capture 70% of the training image data')
print(str(capture80) + ' principal components are needed to capture 80% of the training image data')
print(str(capture90) + ' principal components are needed to capture 90% of the training image data')
print(str(capture95) + ' principal components are needed to capture 95% of the training image data')

storage_required_original = img_size * num_img_train # total entries of original matrix X
storage_required90 = img_size * capture90 + capture90 ** 2 + capture90 * num_img_train # total entries reduced
storage_saved90 = (storage_required_original - storage_required90) / storage_required_original

print('If we only need to recover 90% of the data, we can reduce the storage required by ' + str(f'{storage_saved90:.2%}' + '\n'))

# check shapes of U, S, & V transpose
if np.shape(U) == (img_size, img_size) and np.shape(S) == (num_img_train,) and VT.shape == (num_img_train, num_img_train):
    print('Part 3: Pass - SVD computation completed with correct matrix dimensions.\n')
else:
    print('Part 3: Fail - SVD matrix dimensions do not match expectations.\n')

# check if cumulative variance sums to 1 (within numerical tolerance)
if np.isclose(cumulative_variance[-1], 1.0, atol=1e-6):
    print('Part 3: Pass - Cumulative variance is correctly computed.\n')
else:
    print('Part 3: Fail - Cumulative variance computation may have issues.\n')


# 4 TODO: Limit # of features & plot og image w/ aforementioned features
# DONE

def reconstruct(d):
    U_d = U[:, :d] # 1st d columns of U
    S_d = np.diag(S[:d]) # 1st d singular values
    VT_d = VT[:d, :] # 1st d rows of V transpose
    X_approx = U_d @ S_d @ VT_d # approximation using first d components

    error = np.linalg.norm(X - X_approx, 'fro') ** 2 # error computation
    avg_approx_error = error / num_img_train
    print('The average approximation error for d = ' + str(d) + ' is ' + str(np.rint(avg_approx_error)))

    sample_img_index = 0 # select arbitrary sample image for reconstruction
    reconstructed_img = X_approx[:, sample_img_index].reshape(img_height, img_width)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_train_matrix[:, sample_img_index].reshape(img_height, img_width), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed Image d = ' + str(d))
    plt.savefig('reconstructed' + str(d) + '.png')
    # plt.show()

    return avg_approx_error

error20 = reconstruct(20)
error50 = reconstruct(50)
error70 = reconstruct(70)
error100 = reconstruct(100)
reconstruct(num_img_train) # must be 0 (for sanity check)
print('')

# Note: we were originally unaware that the for loop given in the Jupyter Notebook was supposed to loop thru the 4 given values of d;
# therefore, we initially implemented it differently but will now proceed to put the 4 avg_approx_error values into a list for P/F testing
avg_approx_errors = [error20, error50, error70, error100]
tolerance = 1e-5
if all(earlier >= later - tolerance for earlier, later in zip(avg_approx_errors, avg_approx_errors[1:])):
    print('Part 4: Pass - Approximation error decreases as d increases (within tolerance).\n')
else:
    print('Part 4: Fail - Approximation error does not decrease as expected.\n')


# 5 TODO: Test dataset
# DONE
num_subject_test = 2
num_img_test = num_subject_test * num_entity
img_test_matrix = np.zeros((img_size, num_img_test))

imgs_test = []
for i in os.listdir('./unpadded'):
    if i.startswith('subject14') or i.startswith('subject15'): # last 2 for testing
        imgs_test.append(i)
imgs_test.sort()
# for i in imgs_test:
#     print(i)

for i, j in enumerate(imgs_test):
    face = io.imread(os.path.join('./unpadded', j))
    img_test_matrix[:, i] = face.flatten()
# print(img_test_matrix[0])
# print(np.shape(img_test_matrix))

U50 = U[:, :50] # 1st 50 training features
principal_components_test = np.transpose(U50) @ img_test_matrix
# print(np.shape(principal_components_test))

reconstructed_img_test = U50 @ principal_components_test # reconstruct using previously calculated feature values
error_test = np.linalg.norm(img_test_matrix - reconstructed_img_test, 'fro') ** 2
avg_approx_errors_test = np.mean(error_test)
print('The average approximation error for the test set with 50 features is ' + str(np.rint(avg_approx_errors_test)) + '\n')

sample_img_index_test = 0
reconstructed_img_test_single = reconstructed_img_test[:, sample_img_index_test]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_test_matrix[:, sample_img_index_test].reshape(img_height, img_width), cmap='gray')
plt.title('Original Test Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img_test_single.reshape(img_height, img_width), cmap='gray')
plt.title('Reconstructed Test Image \nw/ 50 Features')
plt.savefig('reconstructed_test.png')
# plt.show()


# 6 TODO: Repeat test w/ rotated image
# DONE
rotated_img = io.imread('subject15rotated.jpeg').flatten()
# print(np.shape(rotated_img))

principal_components_rotated = np.transpose(U50) @ rotated_img
# print(np.shape(principal_components_rotated))

reconstructed_img_rotated = U50 @ principal_components_rotated
error_rotated = np.linalg.norm(rotated_img - reconstructed_img_rotated) ** 2
avg_approx_errors_rotated = np.mean(error_rotated)
print('The average approximation error for the rotated image with 50 features is ' + str(np.rint(avg_approx_errors_rotated)) + '\n')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rotated_img.reshape(img_height, img_width), cmap='gray')
plt.title('Original Rotated Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img_rotated.reshape(img_height, img_width), cmap='gray')
plt.title('Reconstructed Rotated Image \nw/ 50 Features')
plt.savefig('reconstructed_rotated.png')
# plt.show()