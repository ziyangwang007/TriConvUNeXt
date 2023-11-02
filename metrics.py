import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = np.where(output>0.5,1,0)
    target_ = np.where(target>0.5,1,0)
    dice1 = np.sum(target_[output_==1])*2.0 / (np.sum(target_) + np.sum(output_))
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    #====================================================
    true_labels = target_
    pred_labels = output_ 
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    # print('TP:{}, FP:{}, TN:{}, FN:{}'.format(TP,FP,TN,FN))
    TP=float(TP)
    FP=float(FP)
    TN=float(TN)
    FN=float(FN)
   
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    # print('accuracy:{}'.format(accuracy))

    precision = TP/(TP+FP+0.00001)
    # print('precision:{}'.format(precision))

    recall = TP/(TP+FN)
    # print('recall:{}'.format(recall))

    specificity = TN/(TN+FP)

    sentitive = TP/(TP+FN)
    # print('specificity:{}'.format(specificity))
    return iou, dice, accuracy, precision, recall,specificity,dice1,sentitive
    # return iou, dice

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
