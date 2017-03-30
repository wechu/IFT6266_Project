import os, sys
import glob
import pickle as pkl
import numpy as np
import PIL.Image as Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize


def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    ### PATH need to be fixed
    # For training data 
    #data_path="C:/Users/Wes/PycharmProjects/IFT6266Project/inpainting/train2014"
    #save_dir = "C:/Users/Wes/PycharmProjects/IFT6266Project/Tmp/64_64/train2014/"
    # For validation data
    data_path="C:/Users/Wes/PycharmProjects/IFT6266Project/inpainting/val2014"
    save_dir = "C:/Users/Wes/PycharmProjects/IFT6266Project/Tmp/64_64/val2014/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)
    #crop_size = (32, 32)
    print(data_path+"/*.jpg")
    imgs = glob.glob(data_path+"/*.jpg")
    print(imgs)
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print (i, len(imgs), img_path)

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print(tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32))
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))




def show_examples(batch_idx, batch_size,
                  ### PATH need to be fixed
                  mscoco="C:/Users/Wes/PycharmProjects/IFT6266Project/inpainting/", split="train2014", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path,"rb") as fd:
        caption_dict = pkl.load(fd, encoding='ASCII')

    print(data_path + "/*.jpg")
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

    plt.rcParams['toolbar'] = 'None'

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

        #Image.fromarray(img_array).show()
        # Image.fromarray(input, "RGB").show()
        # Image.fromarray(target, "RGB").show()

        print(cap_id)
        plt.figure(figsize=(8,8))
        plt.imshow(input)
        plt.figure(figsize=(4,4))
        plt.imshow(target)
        plt.show()
        print(i, caption_dict[cap_id])



def save_examples(### PATH need to be fixed
                  mscoco="C:/Users/Wes/PycharmProjects/IFT6266Project/inpainting/", split="train2014",
                  caption_path="dict_key_imgID_value_caps_train_and_valid.pkl",
                  save_name="C:/Users/Wes/PycharmProjects/IFT6266Project/Data/train2014.pkl"):
    '''
    Save the pictures as (input, target) pairs
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path, "rb") as fd:
        caption_dict = pkl.load(fd, encoding='ASCII')

    print(data_path + "/*.jpg")
    imgs = glob.glob(data_path + "/*.jpg")

    inputs = []
    targets = []
    inputs_cap = []
    #count = 0
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        #print(img_array.shape)
        if len(img_array.shape) == 3:
            # Note that we ignore the grayscale images (171 of those in training set)
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]

            inputs.append(input)
            targets.append(target)
            inputs_cap.append(caption_dict[cap_id])

            print(i, len(imgs), img_path)
        # else:
            # count += 1
            # print("not right dim", count, img_array.shape)
            # raise ValueError('Picture does not have three dimensions...')

    with open(save_name, 'wb') as pickle_file:
        pkl.dump([inputs, targets, inputs_cap], pickle_file, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    #resize_mscoco()
    #show_examples(11, 3)
    save_examples()
    #save_examples(split="val2014", save_name="C:/Users/Wes/PycharmProjects/IFT6266Project/Data/val2014.pkl")