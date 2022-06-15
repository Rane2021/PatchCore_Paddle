import os
import cv2
import torch
import torchvision
import numpy as np
import onnx
import onnxruntime
from queue import Queue
import shutil
from tqdm import tqdm


# data_path = "/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/172.16.236.123_0608_crop"
data_path = "/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/172.16.236.123_0608_crop_0614_v2"
same_thresh = 0.07  # 0.1


save_path = data_path.replace("crop", "screen_" + str(same_thresh))
if not os.path.exists(save_path):
    os.mkdir(save_path)


# img preprocess
device = torch.device('cuda')
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)
def preprocess(img):
    global device, normalize
    x = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# main
if __name__ == "__main__":
    # load_model
    ort_session = onnxruntime.InferenceSession("trained_models/p_compose_sim.onnx")
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    img_path_list = os.listdir(data_path)
    img_path_list.sort()
    
    
    img_num = 0
    last_frame_outs = 0
    for img_path in tqdm(img_path_list):
        if '.jpg' or '.png' in img_path:
            img_path_ori = os.path.join(data_path, img_path)
            img = cv2.imread(img_path_ori, cv2.IMREAD_COLOR)
            dummy_input = preprocess(img)

            # compute ONNX Runtime output prediction
            onnx_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
            onnx_outs = ort_session.run(None, onnx_inputs)[0][0][40]
            # print("img name: {}; onnx pred max: {}".format(img_path, onnx_outs))
            if abs(last_frame_outs - onnx_outs) > same_thresh:
                save_img_path = os.path.join(save_path, img_path)
                # cv2.imwrite(save_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                shutil.copy(img_path_ori, save_img_path)
                img_num += 1
            
            last_frame_outs = onnx_outs
            
    print("all image num: {}; after screen image num: {}".format(len(img_path_list), img_num))
    print("finish!")


