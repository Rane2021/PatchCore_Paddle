# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import path as osp
import argparse
from time import time
import datetime
import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm

import paddle
from paddle.vision import transforms as T

from paddle import inference
from paddle.inference import Config, create_predictor

from model import PatchCore, postporcess_score_map, get_model
from utils import plot_fig

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    
    
    parser.add_argument("--category", type=str, default='bottle')
    
    parser.add_argument("-i", "--input_file", type=str, help="input file path", 
                         default="/media/tianru/Rane/CODE/04_huagong_proj/03_anomalib/datasets/MVTec/bottle/test/broken_large/006.png")  # --> 006 score：3.466
    # parser.add_argument("-i", "--input_file", type=str, help="input file path", 
    #                     default="/media/tianru/Rane/CODE/04_huagong_proj/03_anomalib/datasets/MVTec/bottle/test/broken_small/006.png")  # --> 006 score：3.08
    # parser.add_argument("-i", "--input_file", type=str, help="input file path", 
    #                      default="/media/tianru/Rane/CODE/04_huagong_proj/03_anomalib/datasets/MVTec/bottle/test/contamination/")  # --> 006 score：2.724  all: 1.85-3.96
    # parser.add_argument("-i", "--input_file", type=str, help="input file path", 
    #                      default="/media/tianru/Rane/CODE/04_huagong_proj/03_anomalib/datasets/MVTec/bottle/test/good/")  # --> 006 score：1.423  all: 1.07-1.6
    
    
    parser.add_argument("--model_name", type=str, default="PatchCore")
    parser.add_argument("--model_file", type=str, default="output/coreset_resnet18_10/infer_model/model.pdmodel")
    parser.add_argument("--params_file", type=str, default="output/coreset_resnet18_10/infer_model/model.pdiparams")
    parser.add_argument("--stats", type=str, default='output/coreset_resnet18_10/stats')
    
    parser.add_argument("--save_path", type=str, default='infer_output/coreset_resnet18_10')
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=4000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=521)

    # params for process control
    parser.add_argument("--enable_post_process", type=str2bool, default=True)
    
    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor

def preprocess(img):
    transform_x = T.Compose([T.Resize(256),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    # x = Image.open(img).convert('RGB')
    x = cv2.imread(img, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    
    x = transform_x(x).unsqueeze(0)
    return x.numpy()


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".png"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files

def postprocess(args, test_imgs, class_name, outputs, stats):
    # 转位Tensor and concat
    outputs = [paddle.to_tensor(i) for i in outputs]
    model = get_model('coreset' if 'memory_bank' in stats.keys() else 'padim+')(None)
    model.load(stats)
    outputs = [model.project(i) for i in outputs]  # no operator
    outputs = paddle.concat(outputs, axis=0)
    
    # calculate score
    score_map, image_score = model.generate_scores_map(outputs, (256, 256))
    print(f'image_score:{image_score}')
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    save_name = args.save_path
    
    plot_fig(test_imgs, scores, None, 0.5, save_name, class_name, True, 'infer')
    print('saved')

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    model_name = args.model_name
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    print('load train set feature from: %s' % args.stats)
    stats = paddle.load(args.stats)

    inference_config, predictor = create_paddle_predictor(args)
    # VIP: 这里爆段错误很大可能是内存问题，切换到GPU就没问题了！

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # Pre process batched input
        batched_inputs = [files[st_idx:ed_idx]]
        imgs = []
        test_imgs = []
        for inp in batched_inputs[0]:
            img = preprocess(inp)
            imgs.append(img)
            test_imgs.extend(img)

        imgs = np.concatenate(imgs)
        batched_inputs = [imgs]

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()

        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        
        
            
        # Rane: test pred one image time:\
        print("start circle test...")
        start = time()
        for i in tqdm(range(100)):
            batched_inputs = [files[st_idx:ed_idx]]
            imgs = []
            test_imgs = []
            for inp in batched_inputs[0]:
                img = preprocess(inp)
                imgs.append(img)
                test_imgs.extend(img)

            imgs = np.concatenate(imgs)
            batched_inputs = [imgs]

            # run inference
            input_names = predictor.get_input_names()
            for i, name in enumerate(input_names):
                input_tensor = predictor.get_input_handle(name)
                input_tensor.reshape(batched_inputs[i].shape)
                input_tensor.copy_from_cpu(batched_inputs[i].copy())

            # do the inference
            predictor.run()
            
            # get out data from output tensor
            results = []
            # get out data from output tensor
            output_names = predictor.get_output_names()
            for i, name in enumerate(output_names):
                output_tensor = predictor.get_output_handle(name)
                output_data = output_tensor.copy_to_cpu()
                results.append(output_data)
        t = time() - start
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Pred ends, total {:.2f}s".format(t))
        
        # 2022-6-2: one image cost: 约500ms
        
        print("finish!")


if __name__ == "__main__":
    main()


