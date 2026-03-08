import os,sys
os.chdir(sys.path[0])

import argparse
import functools
import torch
from modules.model import Model
from data_utils.args_tip import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_path',       str,    'output/models/model.pth',           '模型保存的路径')
add_arg('save_path',        str,    'output/inference/inference.pth',    '模型保存的路径')
args = parser.parse_args()

# 获取模型
model = Model(num_class = 5)
model.load_state_dict(torch.load(args.model_path))
# 加上Softmax函数
model = torch.nn.Sequential(model, torch.nn.Softmax())

# 保存预测模型
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model, args.save_path)
