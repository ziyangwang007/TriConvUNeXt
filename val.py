import argparse
import os
from glob import glob
from torchsummary import summary
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext
from nn.UNetLite import ULite
from nn.InceptionUNext import mynet
from nn.ConvUNext import ConvUNeXt
from nn.UNet import UNet
from nn.ATTUNet import AttU_Net
from nn.SwinUnet import SwinUnet
from nn.SwinUnet import SwinUnet_config
from nn.TransUnet import get_transNet
from nn.ATTUNet import AttU_Net
from nn.DenseUnet import Dense_Unet
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='GLAS_UNext_woDS',
                        help='model name')

    args = parser.parse_args()

    return args
import segmentation_models_pytorch as smp
    model =  UNext()              # model output channels (number of classes in your dataset)
    return model

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = create_model(num_classes=config['num_classes']).cuda()
    model = model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))
    # Data loading code
    if config['dataset'] == 'GLAS':
        val_img_ids = sorted(glob(os.path.join(config['dataset'], 'test', 'images', '*' )))
        val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    else:
        img_ids = sorted(glob(os.path.join(config['dataset'], 'images', '*' + config['img_ext'])))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)

    # model.load_state_dict(torch.load('models/%s/model.pth' %
    #                            config['name']),map_location='cuda:0')
    
    loaded_state = torch.load('models/%s/model.pth' %
                               config['name'], map_location='cuda:0')
    model.load_state_dict(loaded_state)

    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        # Resize(224, 224),
        transforms.Normalize(),
    ])
    if config['dataset'] == 'GLAS':
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(config['dataset'], 'test','images'),
            mask_dir=os.path.join(config['dataset'], 'test','masks'),
            img_ext='.bmp',
            mask_ext='.bmp',
            num_classes=config['num_classes'],
            transform=val_transform)
    else:
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(config['dataset'], 'images'),
            mask_dir=os.path.join(config['dataset'], 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    acc_avg_meter = AverageMeter()
    pr_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    sp_avg_meter = AverageMeter()
    dice1_avg_meter = AverageMeter()
    se_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()
    count_list = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)


            iou,dice, accuracy, precision, recall,specificity,dice1,sentitive = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            acc_avg_meter.update(accuracy, input.size(0))
            pr_avg_meter.update(precision, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            sp_avg_meter.update(specificity, input.size(0))
            dice1_avg_meter.update(dice1, input.size(0))
            se_avg_meter.update(sentitive, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))
    # print(len(pr_list))
    import pandas as pd


    print('IoU: %.5f' % iou_avg_meter.avg)
    print('Dice: %.5f' % dice_avg_meter.avg)
    print('ACC: %.5f' % acc_avg_meter.avg)
    print('PR: %.5f' % pr_avg_meter.avg)
    # print('RECALL: %.5f' % recall_avg_meter.avg)
    print('SE: %.5f' % se_avg_meter.avg)
    print('SP: %.5f' % sp_avg_meter.avg)
    # print('SE: %.5f' % se_avg_meter.avg)
    print('Dice1: %.5f' % dice1_avg_meter.avg)
    print('RECALL: %.5f' % recall_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
