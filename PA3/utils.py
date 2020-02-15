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
        ints.append(intersection)
        unions.append(union)

    return ints,unions


def pixel_acc2(pred, target):
    f = torch.zeros(target.shape).cuda()
    t = torch.ones(target.shape).cuda()
    mask = torch.where((target == 0) | (target == 1) | (target == 2) | (target == 3) | (target == 4) |(target == 5) |(target == 6) |(target == 9) |(target == 10) |(target == 14) |(target == 15) |(target == 16) | (target == 18) |(target == 29) |(target == 30), f, t)
    target = target[mask == 1.0]
    pred = pred[mask == 1.0]
    total_pixels = target.numel()
    return torch.tensor(torch.sum(pred.eq(target)).item()/total_pixels)

def one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(nn.Module):        
    def forward(self, output, target):
        target = one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + 1) /
                    (output_flat.sum() + target_flat.sum() + 1))
        return loss
>>>>>>> Stashed changes
