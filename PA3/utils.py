import torch
from dataloader import labels_classes
import torch.nn as nn
import torch.nn.functional as F


def iou(pred, target,n_class):
    """
    Calulates the intersection over union between prediction and target.
    Formula--> iou for class c = number of overlapping pixels / number of pixels of in class c in both prediction and target

    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: List of iou accuracies where index corresponds to class label.
    """
    ious = []
    for c in range(n_class):
        # Converting pixels to T/F for class "c"
        pred_class_n = pred == c #type bool
        target_class_n = target == c #type bool
        
        # Intersection calculation is simply AND between tensors
        intersection = (pred_class_n & target_class_n).sum().double()
        #Union calculation is simpy OR between tensors
        union = (pred_class_n + target_class_n).sum().double()

        if union == 0:
            ious.append(float('nan'))
            # ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            iou_for_class_c = intersection/union
            ious.append(iou_for_class_c.item()) # Append the calculated IoU to the list ious

    return ious


def pixel_acc(pred, target):
    """
    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: Accuracy pixel wise
    """
    num_pixels = float(pred.shape[0] * pred.shape[1] * pred.shape[2])
    return torch.sum(pred.eq(target))/num_pixels

def iou2(pred, target,n_class):
    """
    Calulates the intersection over union between prediction and target.
    Formula--> iou for class c = number of overlapping pixels / number of pixels of in class c in both prediction and target

    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: List of iou accuracies where index corresponds to class label.
    """
    ints = []
    unions = []
    for c in range(n_class):
        if(labels_classes[c].ignoreInEval):
            continue
        # Converting pixels to T/F for class "c"
        pred_class_n = pred == c #type bool
        target_class_n = target == c #type bool
        
        # Intersection calculation is simply AND between tensors
        intersection = (pred_class_n & target_class_n).sum().double()
        #Union calculation is simpy OR between tensors
        union = (pred_class_n + target_class_n).sum().double()
        ints.append(intersection.item())
        unions.append(union.item())

    return ints,unions


def pixel_acc2(pred, target):
    f = torch.zeros(target.shape).cuda()
    t = torch.ones(target.shape).cuda()
    mask = torch.where((target == 0) | (target == 1) | (target == 2) | (target == 3) | (target == 4) |(target == 5) |(target == 6) |(target == 9) |(target == 10) |(target == 14) |(target == 15) |(target == 16) | (target == 18) |(target == 29) |(target == 30), f, t)
    target = target[mask == 1.0]
    pred = pred[mask == 1.0]
    total_pixels = target.numel() + .000000000000001
    return torch.tensor(torch.sum(pred.eq(target)).item()/total_pixels)

def one_hot(labels, num_classes):
    one_hot_vec = torch.cuda.FloatTensor(
        labels.size()[0], 
        num_classes, 
        labels.size()[2], 
        labels.size()[3]).zero_()
    t = one_hot_vec.scatter_(1, labels.data, 1)
    return t

class DiceLoss(nn.Module):        
    def forward(self, op, t):
        op = F.softmax(op, dim=1)
        op_f = op.contiguous().view(-1)
        t = one_hot(t.unsqueeze(dim=1), num_classes=op.size()[1])
        t_f = t.contiguous().view(-1)
        loss = 1 - ((2. * (op_f * t_f).sum() + 1) / (op_f.sum() + t_f.sum() + 1))
        return loss.mean()
    
class GDiceLoss(nn.Module):
    def forward(self, op, t):
        t = one_hot(t.unsqueeze(dim=1), num_classes=op.size()[1])
        op = torch.sigmoid(op)
        n = torch.sum((op*t), dim=(2, 3))
        d = torch.sum(op.pow(1)+t.pow(1), dim=(2, 3))
        n = torch.sum(n, dim=1)
        d = torch.sum(d, dim=1)
        loss = 1 - (2*n+1)/(d+1)
        return loss.mean()
