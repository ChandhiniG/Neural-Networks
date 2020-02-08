def iou(pred, target):
    """
    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: IOU accuracy
    """
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = 0# intersection calculation
        union = 0#Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious = []# Append the calculated IoU to the list ious
    return ious


def pixel_acc(pred, target):
    """
    Pred: Tensor (B x W x H)
    Target: Tensor (B x W x H)
    Returns: Accuracy pixel wise
    """
    num_pixels = float(pred.shape[0] * pred.shape[1])
    return torch.sum(pred.eq(target))/num_pixels