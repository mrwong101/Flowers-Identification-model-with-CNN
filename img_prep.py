import cv2
import os
from os import listdir

# Renaming files
def rename(path):
    files = listdir(path)
    i=0
    for j in files:
        for f in files:
            try:
                src =path+'\\'+ f
                dst =path+'\\'+ str(i)+ ".jpg"
                os.rename(src, dst)
            except:
                pass
            i += 1
   
for p in range(3):
    path = ".\\img3\\" +str(p)
    rename(path)

# This part is related to data resizing and augmentation to increase sample by 2x
pic_size = 50
p = 2
path = ".\\img3\\" +str(p)
resized_path = ".\\img3\\resized_"+str(p)
files = listdir(path)

i=0
for j in files:
    img = cv2.imread(path+ '\\'+ j, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    
#    Resized with resolution decrease
    dim = (pic_size, pic_size)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    cv2.imwrite(resized_path+"\\"+str(i)+".jpg", resized)
    i += 1
    
#    Resized with just centered part
    center1 = int((img.shape[0]-pic_size)/2)
    center2 = int((img.shape[1]-pic_size)/2)

    step = int(pic_size)
    crop_img = img[center1-step:center1+step, center2-step:center2+step]
    resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(resized_path+"\\"+str(i)+".jpg", resized)
    i += 1

# This part is related to creating the npz file
    
import numpy as np
import imageio

num_imgs_for_training = 80
num_samples = 3
num_imgs_per_sample = 100

num_total = num_samples * num_imgs_per_sample
num_total_train = num_samples * num_imgs_for_training

img_width = 50
img_height = 50
img_channel = 3

scene_data = []
scene_labels = []

for i in range(num_samples):
    path = './img3/resized_' + str(i) + '/'
    files = listdir(path)
    for j in files:
        print(path+"/"+j)
        img = imageio.imread(path+"/"+j)
        img_1D = np.reshape(img, (img_width*img_height*img_channel))/255
        img_1D_list = np.asarray(img_1D).tolist()
        # Keep Three Decinalnp.asarray(scene_data1)
        img_1D_scaled = [np.float32(x) for x in img_1D_list]
        scene_data.append(img_1D_scaled)
        scene_labels.append(i)

scene_data_full = np.asarray(scene_data)
scene_labels_full = np.asarray(scene_labels)
print("The images have been imported successfully!")

index_random = np.random.permutation(len(scene_data_full))
scene_data_random = scene_data_full[index_random]
scene_labels_random = scene_labels_full[index_random]

train_index = [i for i in range(num_total_train)]
train_data = scene_data_random[:num_total_train]
train_labels = scene_labels_random[:num_total_train]
eval_index = [i for i in range(num_total_train,num_total)]
eval_data = scene_data_random[num_total_train:]
eval_labels = scene_labels_random[num_total_train:]

np.savez('hkflowers.npz',
         train_data = train_data, train_labels = train_labels,
         eval_data = eval_data, eval_labels = eval_labels,
         train_index = train_index, eval_index = eval_index)

