import torch
import os


def separate_and_save_weights(model_path, target_dir, backbone_prefix='extractor', convert_prefix='convert', attention_prefix='attention', vit_prefix='vit', head_prefix='head'):
   full_model_state = torch.load(model_path, map_location='cpu')
   
   backbone_weights = {}
   vit_weights = {}
   head_weights = {}
   convert_weights = {}
   attention_weights = {}
   
   if 'state_dict' in full_model_state:
       full_model_state = full_model_state['state_dict']
   
   for key, value in full_model_state.items():
       if key.startswith(backbone_prefix):
           new_key = key[len(backbone_prefix)+1:]  # +1 to remove the trailing dot as well
           backbone_weights[new_key] = value
       elif key.startswith(vit_prefix):
           new_key = key[len(vit_prefix)+1:]  # +1 to remove the trailing dot as well
           vit_weights[new_key] = value
       elif key.startswith(head_prefix):
           new_key = key[len(head_prefix)+1:]  # +1 to remove the trailing dot as well
           head_weights[new_key] = value
       elif key.startswith(convert_prefix):
           new_key = key[len(convert_prefix)+1:]  # +1 to remove the trailing dot as well
           convert_weights[new_key] = value
       elif key.startswith(attention_prefix):
           new_key = key[len(attention_prefix)+1:]  # +1 to remove the trailing dot as well
           attention_weights[new_key] = value
   
   if len(backbone_weights)>0:
       torch.save(backbone_weights, f'{os.path.join(target_dir, "backbone_weights.pth")}')
   
   if len(vit_weights)>0:
       torch.save(vit_weights, f'{os.path.join(target_dir, "vit_weights.pth")}')
   
   if len(head_weights)>0:
       torch.save(head_weights, f'{os.path.join(target_dir, "head_weights.pth")}')
   
   if len(convert_weights)>0:
       torch.save(convert_weights, f'{os.path.join(target_dir, "convert_weights.pth")}')
   
   if len(attention_weights)>0:
       torch.save(attention_weights, f'{os.path.join(target_dir, "attention_weights.pth")}')
   
   print('Weights have been separated and saved.')


import sys


# args = sys.argv[1:]


# model_path = args[0]
# target_dir = args[1]


separate_and_save_weights("C:\\Users\\Valentin\\Desktop\\github\\seminar-2-heads-global-pool\\seminar-big-three\\APViT-main\\epoch_20.pth", "C:\\Users\\Valentin\\Desktop\\github\\seminar-2-heads-global-pool\\seminar-big-three\\APViT-main\\")