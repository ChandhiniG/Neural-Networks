import torch
from dataloader import labels_classes

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
    ious = []
    tot_int = 0
    tot_union = 0
    for c in range(n_class):
        if(labels_classes[c].ignoreInEval):
            ious.append(float('nan'))
            continue
        # Converting pixels to T/F for class "c"
        pred_class_n = pred == c #type bool
        target_class_n = target == c #type bool
        
        # Intersection calculation is simply AND between tensors
        intersection = (pred_class_n & target_class_n).sum().double()
        #Union calculation is simpy OR between tensors
        union = (pred_class_n + target_class_n).sum().double()
        tot_int += intersection
        tot_union += union
        if union == 0:
            ious.append(float('nan'))
            # ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            iou_for_class_c = intersection/union
            ious.append(iou_for_class_c.item()) # Append the calculated IoU to the list ious

    return ious,tot_int/tot_union


def pixel_acc2(pred, target):
    """
    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: Accuracy pixel wise
    """
    num_pixels = float(pred.shape[0] * pred.shape[1] * pred.shape[2])
    return torch.sum(pred.eq(target))/num_pixels
