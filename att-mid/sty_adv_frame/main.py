# targetd attack, target is chosen randomly
import argparse
import os
import time
from adv_trainer_tar import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Style Adversarial bordering')
    parser.add_argument('--width', '-w', default=4, type=int, help='Width of the bordering')
    parser.add_argument('--keep-size', action='store_true',
                        help='If set, image will be rescaled before applying bordering, so that unattacked and attacked '
                             'images have the same size.')
    parser.add_argument('--target', type=int, default=1, 
                        help='Target class. If unspecified, untargeted attack will be performed. If set to -1, '
                             'target will be chosen randomly. Note that in targeted attack we aim for higher accuracy '
                             'while in untargeted attack we aim for lower accuracy.')
    parser.add_argument('--epochs', default=1,  type=int)
    parser.add_argument('--batch-size', default=1,  type=int) # batch-size设置为1，加载单张图片
    parser.add_argument('--lr', default=1, type=float, help='Initial learning rate for the bordering')
    parser.add_argument('--lr-decay-wait', default=1, type=int, help='How often (in epochs) is lr being decayed')
    parser.add_argument('--lr-decay-coefficient', default=0.98, type=float,
                        help='When learning rate is being decayed, it is multiplied by this number')
    parser.add_argument('--print-freq', default=20, type=int, help='Print frequency (in batches)')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loading workers for, each of train and eval data will get that many')
    
    parser.add_argument('--dataset_path', default='../dataset/all_kinds')
    parser.add_argument('--bkground_path', default='../dataset/bkgrounds/') #bkgrounds/)
    parser.add_argument('--border_chk_save_path', default='../results/adv_border_chk/')
    parser.add_argument('--border_png_save_path', default='../results/adv_border_png/')
    parser.add_argument('--adv_save_path', default='../results/adv_img/')
    parser.add_argument('--start_index', default=0) #从哪张图片开始处理
    
    args = parser.parse_known_args()[0]

    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_time = time.time()
    Trainer(args).train(args.epochs)
    end_time = time.time()
    print("Total training time(minute):", (end_time-start_time)/60)

