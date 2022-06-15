import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO: camera crop param
# 0608
# cam_123 = [418, 140, 2055, 1013]
# data_path = "/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/172.16.236.123_0608"
# 0614
# cam_123 = [1496, 82, 2065, 1022]
# data_path = "/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/172.16.236.123_0608"

# 0614 V2
cam_123 = [1742, 186, 2372, 783]
data_path = "/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/172.16.236.123_0608_manual_select"


# img = data_path + "/20220525145231.jpg"
# img = cv2.imread(img, cv2.IMREAD_COLOR)
# plt.imshow(img)
# plt.show()


save_path = data_path + "_crop_0614"
if not os.path.exists(save_path):
    os.mkdir(save_path)


# main
if __name__ == "__main__":

    img_path_list = os.listdir(data_path)
    img_path_list.sort()    
    
    for img_path in tqdm(img_path_list):
        if '.jpg' in img_path:
            img = os.path.join(data_path, img_path)
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img = img[cam_123[1]:cam_123[3], cam_123[0]:cam_123[2], :]
            # plt.imshow(img)
            # plt.show()   
            img_path = img_path.replace(".jpg", ".png")
            save_img_path = os.path.join(save_path, img_path)
            # cv2.imwrite(save_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])     
            cv2.imwrite(save_img_path, img)      

    print("crop finish!")


