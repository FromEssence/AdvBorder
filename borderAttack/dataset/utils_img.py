

# 给定图片，截取指定长度的边框 拼在一起显示
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import copy
import os

def resize_img(imgPath, aimShape):
    im = Image.open(imgPath).convert('RGB')
    im = im.resize(aimShape, Image.ANTIALIAS)
    im.save('./Images/232_grass_dog.png')

def getStyleImg(imgPath, savePath, aimShape=(224,224)):
    '''
    缩放到aimShape-->裁剪四周作为风格图片和内容图片
    如果时灰度图像，返回False
    '''

    im = Image.open(imgPath).convert('RGB')
    im = im.resize(aimShape, Image.ANTIALIAS)

    im_np = np.array(im)

    if len(im_np.shape)!=3:
        return False

    width_border = aimShape[0]-4
    bk_border = np.zeros((32, width_border, 3))
    bk_border[:4, :, :] = im_np[:4, :width_border, :]
    bk_border[4:8, :, :] = im_np[:width_border, width_border:aimShape[0], :].transpose(1,0,2)
    bk_border[8:12, :, :] = im_np[width_border:aimShape[0], 4:aimShape[0], :]
    bk_border[12:16, :, :] = im_np[4:aimShape[0], :4, :].transpose(1,0,2)
    bk_border[16:, :, :] = copy.deepcopy(bk_border[:16, :, :])


    bk_border = Image.fromarray(bk_border.astype(np.uint8))
    bk_border.save(savePath)
    
    return True

def get_dog_label():
    imgPath = 'imgnet_set/dogs_outer/dogs'
    img_names = os.listdir(imgPath)
    with open('dog_1000_lable.txt', 'wt') as f:
        
        for i in range(1000):
            f.write(img_names[i][6:9])
            f.write('\n')


def get_advs(border_paths = './results/nostyle_adv_v0/', img_paths = 'E:/conda_projs/pytorch-ssim-master/ori/dogs/', save_path='./nostyleadvs/'):
    '''
     将原图和训练过的边框拼接后保存
    '''
    # border_paths = './results/nostyle_adv_v0/'
    # img_paths = 'E:/conda_projs/pytorch-ssim-master/ori/dogs/'
    
    imgnames = os.listdir(img_paths)
    bordernames = os.listdir(border_paths)
    for i in range(len(bordernames)):
        border = Image.open(border_paths+bordernames[i]).convert('RGB')
        img = Image.open(img_paths+imgnames[i]).convert('RGB').resize((224,224), Image.ANTIALIAS)

        border = np.array(border)
        img = np.array(img)
        width = 4
        img[:width, :-width,:] = border[0:4,:,:]
        img[:-width, -width:,:] = border[4:8,:,:].reshape((220,4,3))
        img[-width:, width:,:] = border[8:12,:,:]
        img[width:, :width,:] = border[12:16,:,:].reshape((220,4,3))

        bk_border = Image.fromarray(img.astype(np.uint8))
        bk_border.save(save_path+imgnames[i])
if __name__ == "__main__":
    get_advs(border_paths='./results/mysty_content/', save_path='./my_style_content/')
    

    