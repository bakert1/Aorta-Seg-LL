import matplotlib.pyplot as plt
import monai
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

HEATMAP_KERNELS = ["guassian"]
CENTERS_POSTFIX = "landmarks"
NUM_LANDMARKS = 6

##########################
# Segmentation Utilities #
##########################
class BoundaryFromSeg:
    """
    Defines a 3D Sobel edge detection operator which can be used to find a segmentation's boundary.
    The operator has 3 kernels, one for each direction (x, y, z). Each kernel is a 3x3x3 matrix. The x kernel
    detects edges in the x direction. Similarly, the y and z kernels detect edges in the y and z directions.

    For more details about 3 dimensional Sobel edge detection operator, see the two answers to this question.
       - https://stackoverflow.com/questions/7330746/implement-3d-sobel-operator
    See also the Wikipedia page:
       - https://en.wikipedia.org/wiki/Sobel_operator
    """
    def __init__(self, threshold=0.2):
        # Start by defining 3x3 matrices that will be used to build the 3x3x3 kernels.
        x = -np.array([[1., 0, -1], [2, 0, -2], [1, 0, -1]])
        y = np.transpose(x)
        z = np.array([[1., 2, 1], [2, 4, 2], [1, 2, 1]])

        #  Build the 3x3x3 kernels from the matrices
        x = np.concatenate([[x], [2*x], [x]]) / 32.
        y = np.concatenate([[y], [2*y], [y]]) / 32.
        z = np.concatenate([[z], [0*z], [-z]]) / 32.

        # Construct the operator.
        self.operator = torch.tensor(np.array([z, y, x])).unsqueeze(1)
        self.threshold = threshold

    def obtain_boundary(self, x):
        """
        Find the boundary of a set of segmentation images x by applying a Sobel edge detector. Produces a set of
        boundary images 'b' where b[i][j,k,l] = 1 means that x[i][0][j,k,l] is a segmentation boundary voxel.
        Args:
            x: Set of 3D segmentation images. Should have shape (batch_size, channels=1, height, width, depth).

        Returns: a boundary image with shape (batch_size, height, width, depth).
        Raises:
            Assertion Error: If x.shape[1] is not 1.
        """
        assert x.shape[1] == 1 and x.ndim == 5

        self.operator = self.operator.to(x)
        # Compute the image gradient using the Sobel operator.
        # A padding of 1 ensures the resulting boundary mask has the same dimension as input image x.
        # F.conv3d can only pad the image with 0s, so we use F.pad to specify a pad that replicates the
        # value at image edge. Replication padding seems more appropriate for edge detection than zero padding.
        grad = F.pad(x, pad=tuple(1 for _ in range(6)), mode='replicate')
        grad = F.conv3d(grad, self.operator)

        # A pixel is considered to an edge if the gradient magnitude exceeds some threshold value.
        bound = grad.square().sum(dim=1).sqrt() > self.threshold
        return bound


def get_largest_component(segmentation):
    """
    TODO: Replace all usages of this function with MONAI's implementation. Then remove this function.
    Get the largest connected component in a given segmentation.
    This function is useful because the U-Net segmentation sometimes picks up random noise from around the scan.
    We assume the largest connected component of the segmentation is the aorta.

    :param segmentation: a segmentation
    :return: a boolean mask representing the largest connected component in the given segmentation.
    """
    is_tensor = isinstance(segmentation, torch.Tensor)  # bool that is used twice in this function
    if is_tensor:
        device = segmentation.device
        segmentation = segmentation.cpu().numpy()
    # First use scipy to extract the connected components of the segmentation. Each entry in labeled_array is an
    # integer corresponding to the class (segmentation component) that the corresponding voxel belongs to.
    labeled_array, num_features = scipy.ndimage.measurements.label(segmentation)

    # find the largest component, but ignore the background component
    component_sizes = [np.count_nonzero(labeled_array == label_idx) for label_idx in range(1, num_features + 1)]
    if len(component_sizes) > 0:
        largest_component_idx = np.argmax(component_sizes) + 1  # +1 because we ignore the background component
    else:
        # no components were found. This may happen early in training when segmentation is all 0s.
        largest_component_idx = -1

    # return a boolean mask representing the largest connected component of the segmentation
    if is_tensor:
        labeled_array = torch.from_numpy(labeled_array)
        out = torch.zeros(segmentation.shape, device=device)
    else:
        out = np.zeros_like(segmentation)

    out[labeled_array == largest_component_idx] = 1

    return out


def get_seg_criterion(name, loss_params={}):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        # DiceLoss needs to be used with sigmoid=True because our network outputs are logits
        return monai.losses.DiceLoss(sigmoid=True, **loss_params)
    elif name == "focal":
        return monai.losses.FocalLoss(**loss_params)
    elif name == "dicefocal":
        return monai.losses.DiceFocalLoss(**loss_params)
    else:
        raise NotImplementedError("Loss not found.")


######################
# Landmark Utilities #
######################
def multidim_argmax(zs):
    """Pytorch/Numpy argmax function are not flexible enough to find the argmax of every volume along an axis (e.g.,
    they can't be used to find the argmax of each of the 6 heatmaps). This method implements that functionality."""
    # zs.shape is (N=1, L, H, W, D)
    # N: batch size, L: number of landmarks, H: img_height, W: img width, D: img depth
    assert zs.shape[0] == 1

    # Grab the one set of landmark heatmaps
    z = zs[0]

    # For each landmark hmap, find the coordinate of the landmark center
    centers = torch.argmax(torch.reshape(z, (z.shape[0], -1)), dim=-1)

    # torch.argmax returns the flattened index, so we need to unravel it back into a 3D index
    centers_z = centers % z.shape[3]
    centers_y = (centers//z.shape[3]) % z.shape[2]
    centers_x = (centers//z.shape[3]//z.shape[2]) % z.shape[1]

    # axis=-1 is important to ensure correct shape and unsqueeze(0) adds back the batch dimension taken out earlier
    return torch.stack([centers_x, centers_y, centers_z], axis=-1).unsqueeze(0)


def WeightedBCELogitsLoss(weight, reduce=True):
    def loss(input, target):
        bce = - weight * target * F.logsigmoid(input) - (1 - target) * (1 - weight) * F.logsigmoid(-input)
        if reduce:
            return torch.mean(bce)
        else:
            return bce
    return loss


def get_heatmap_criterion(hmap_loss, wbce_pos_weight):
    if hmap_loss == "wbce":
        hmap_criterion = WeightedBCELogitsLoss(weight=wbce_pos_weight, reduce=False)
    elif hmap_loss == "mse":
        hmap_criterion = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError("Loss not found.")
    return hmap_criterion


def get_heatmap_filename(kernel, scale, clip):
    if kernel == "gaussian":
        clip_str = int(1e5*clip)
        scale_str = int(1e2*scale)
        filename = f"heatmap_{kernel}_s{scale_str}_c{clip_str}.nii.gz"
    else:
        raise ValueError(f"Heatmap kernel: {kernel} is invalid. Please choose one of {HEATMAP_KERNELS}.")

    return filename


###########
# Metrics #
###########
def get_confusion(prediction, target):
    lib = torch if type(prediction) is torch.Tensor else np
    prediction_neg = ~prediction
    target_neg = ~target
    tp = lib.sum(prediction*target)
    fp = lib.sum(prediction*target_neg)
    fn = lib.sum(prediction_neg*target)
    tn = lib.sum(prediction_neg*target_neg)

    return tp, fp, fn, tn


def get_dice(segmentation, target):
    # for difference between DICE and IOU: https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    tp, fp, fn, tn = get_confusion(segmentation, target)
    dice = 2 * tp / (2 * tp + fp + fn)  # Dice score. Shape: (1,)
    return dice


def get_iou(segmentation, target):
    # for difference between DICE and IOU: https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    tp, fp, fn, tn = get_confusion(segmentation, target)
    iou = tp / (tp + fp + fn)
    return iou


def get_dice_and_iou(segmentation, target):
    # for difference between DICE and IOU: https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    tp, fp, fn, tn = get_confusion(segmentation, target)
    dice = 2 * tp / (2 * tp + fp + fn)  # Dice score. Shape: (1,)
    iou = tp / (tp + fp + fn)
    return dice, iou


def distance(x, y, pixdim=None):
    """
    Calculates the Euclidean distances between two inputs.
    """

    if pixdim is None:
        return (x-y).square().sum(axis=-1).sqrt()
    else:
        # useful for calculating distances of up-sampled images (non-isotropic spacing)
        print("Warning: distance(x, y, pixdim) with pixdim argument is not tested!")
        pixdim = pixdim.unsqueeze(0).unsqueeze(0)
        res = pixdim * (x - y)
        return torch.sum(res**2, axis=-1)**(1/2)


############
# Plotting #
############
def create_flat_seg_with_landmarks(seg, ll_coords, flip=True, filename=None, show=True):
    planes = ["sagital", "coronal", "axial"]
    seg = seg.cpu().numpy().squeeze()

    # set up segmentation
    seg = seg > 0.5

    transposes = [True, True, True]

    hmap_coords = []
    for ll_idx, ll in enumerate(ll_coords):
        # set up heatmap
        x, y, z = ll
        hmap_coords.append([(y, z), (x, z), (x, y)])

    hmap_coords = np.array(hmap_coords)
    for idx in range(3):
        curr_seg = seg.mean(axis=idx)
        if transposes[idx]:
            curr_seg = curr_seg.T
        mask_seg = np.ma.masked_where(curr_seg <= 1e-4, curr_seg)
        origin = "lower" if flip else "upper"
        plt.imshow(mask_seg, cmap='Reds', vmin=-curr_seg.max(), vmax=curr_seg.max()*1.2, origin=origin)
        for l_idx in range(6):
            star = hmap_coords[l_idx][idx]
            if l_idx == 0:
                plt.scatter(star[0], star[1], marker="*", s=100, color='blue', label="Landmark")
            else:
                plt.scatter(star[0], star[1], marker="*", s=100, color='blue')

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.legend(fontsize=16, loc='upper left')
        plt.tight_layout()
        plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, hspace=0.2, wspace=0.2)
        if filename is not None:
            plt.savefig(f"{filename}_{planes[idx]}.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.clf()



