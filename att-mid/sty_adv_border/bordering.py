# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import copy
import os
from PIL import Image

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
# 四周背景图片位置
imgnames = os.listdir('../dataset/bkgrounds')
# 测试两张
#imgnames = os.listdir('../dataset/few_bk')
imgnames.sort()
loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image)
    return image.to(device, torch.float)


class bordering(nn.Module):
    def __init__(self, width, image_side_length, normalize, scale, img_id, keep_size=True):
        '''
            img_id是当前处理的图像id
        '''
        super(bordering, self).__init__()

        self.width = width
        # If keep_size is True, unattacked and attacked images will have the same size.
        self.keep_size = keep_size
        if keep_size:
            self.length = image_side_length - width
        else:
            self.length = image_side_length + width

        self.attack_shape = [3, 32, 220, 1]
        # print(self.attack_shape)
        # 使用相应背景初始边框
        border_init = torch.randn(self.attack_shape) 
        border_init[:,:,:,0] = image_loader('../dataset/bkgrounds/'+imgnames[img_id]) 

        self.attack = nn.Parameter(border_init)
        
        self.normalize = normalize
        self.scale = 1 #传入默认为1

    def forward(self, input, do_normalize=True):
        self.attack.data = self.attack.clamp(0, 1)
        attack = self.attack
        if do_normalize:
            attack = self.normalize(attack)

        input_size = input.size()
        attacked_size = list(input_size)
        if not self.keep_size:
            attacked_size[-2] = self.length + self.width
            attacked_size[-1] = self.length + self.width

        if len(input_size) == 5:
            # swap color and time dimensions, merge batch and time dimensions
            input = input.permute(0, 2, 1, 3, 4).contiguous().view(-1, input_size[1], input_size[3], input_size[4])

        attacked_size_merged = list(input.size())
        if not self.keep_size:
            attacked_size_merged[-2] = self.length + self.width
            attacked_size_merged[-1] = self.length + self.width

        inner = input

        # attack middle
        borderd_input = copy.deepcopy(inner)
        borderd_input[0, :, 160:164, :220] = attack[:, 0:4, :, 0] # [channal, height, width, batch]
        borderd_input[0, :, 164:168, :220] = attack[:, 8:12, :, 0]
        borderd_input[0, :, 168:172, :220] = attack[:, 16:20, :, 0]
        borderd_input[0, :, 172:176, :220] = attack[:, 24:28, :, 0]


        if len(input_size) == 5:
            borderd_input = borderd_input.view(attacked_size[0], attacked_size[2], attacked_size[1], attacked_size[3],
                                             attacked_size[4])
            borderd_input = borderd_input.permute(0, 2, 1, 3, 4)

        return borderd_input, attack
