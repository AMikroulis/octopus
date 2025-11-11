
import os
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy as scy
import skimage as skim
import scipy.ndimage as ndi
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from scipy.ndimage import distance_transform_edt
from skimage import filters
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from scipy import ndimage
from multiprocessing import Pool, cpu_count

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.simplefilter("ignore", category=CryptographyDeprecationWarning)


def create_segmented_image(image, segmentation_mask):
    # first, let's keep only the largest connected component in the segmentation mask
    segmentation_mask = ndi.binary_opening(segmentation_mask, structure=npy.ones((3,3)))
    labeled_mask, num_features = ndimage.label(segmentation_mask)
    if num_features > 1:
        sizes = ndimage.sum(segmentation_mask, labeled_mask, range(num_features + 1))
        max_label = npy.argmax(sizes[1:]) + 1
        segmentation_mask = (labeled_mask == max_label)

    segmented_image = npy.zeros(image.shape, dtype=npy.uint8)
    non_shifted_segmented_image = npy.zeros(image.shape, dtype=npy.uint8)
    max_height = 0

    for col_idx in range(image.shape[1]):
        mask_column = segmentation_mask[:, col_idx]
        image_column = image[:, col_idx]
        
        # find the first and last True value in the mask
        true_indices = npy.where(mask_column)[0]
        
        if len(true_indices) > 0:
            first_true = true_indices[0]
            last_true = true_indices[-1]
            
            # extract the corresponding image values
            extracted_values = image_column[first_true:last_true+1]
            
            # place the extracted values at the beginning of the column
            segmented_image[:len(extracted_values), col_idx] = extracted_values

            # place the extracted values at their original position for non-shifted image
            non_shifted_segmented_image[first_true:last_true+1, col_idx] = extracted_values
            
            
            # update max_height for plotting purposes
            max_height = max(max_height, len(extracted_values))

    return segmented_image, non_shifted_segmented_image, max_height

def subset_lumi(source, low=0, high=255):
    thresholded = (source >= low) & (source < high)
    return thresholded


def process_frame(args):
    frame, image, thresholds = args
    cv = chan_vese(
        image,
        mu=0.25,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=50,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=True,
    )

    opened = ndi.binary_opening(image, structure=npy.ones((3,3)))
    shifted, segmented_image, max_height = create_segmented_image(image, opened)

    padded = npy.pad(segmented_image>0, pad_width=15, mode='constant', constant_values=0)
    blurred = ndi.distance_transform_edt(padded)
    blurred_neg = ndi.distance_transform_edt(1-padded)
    blurred_sum = (blurred - blurred_neg) >-15
    blurred_reduced = ndi.binary_erosion(blurred_sum, structure=npy.ones((23,23)))
    blurred_reduced_no_padding = blurred_reduced[15:-15,15:-15]
    segmented_region = blurred_reduced_no_padding * image

    regions = npy.digitize(segmented_region, bins=thresholds)
    regions_colorized = skim.color.label2rgb(regions)

    colour_dict = [0, 1, 100, 101, 110]
    regions_recolour = npy.zeros((segmented_region.shape[0], segmented_region.shape[1]), 'int64')
    for x in range(segmented_region.shape[0]):
        for y in range(segmented_region.shape[1]):
            colour = 100*regions_colorized[x,y,0]+10*regions_colorized[x,y,1]+1*regions_colorized[x,y,2]
            regions_recolour[x,y] = colour_dict.index(colour)

    new_image = npy.zeros(segmented_region.shape)
    for col_idx in range(image.shape[1]):
        mask_column = blurred_reduced_no_padding[:, col_idx]
        image_column = regions_recolour[:, col_idx]
        true_indices = npy.where(mask_column)[0]
        if len(true_indices) > 0:
            first_true, last_true = true_indices[0], true_indices[-1]
            extracted_values = image_column[first_true:last_true+1]
            new_image[:len(extracted_values), col_idx] = extracted_values

    return new_image

