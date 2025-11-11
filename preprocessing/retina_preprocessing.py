import numpy as npy
import pandas as pd
import os
import warnings
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial
import scipy as scy
import skimage as skim
from joblib import load
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.ndimage as ndi
from scipy.spatial.distance import euclidean
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import chan_vese
from scipy.spatial import KDTree
import pickle
import matplotlib.patches as patches
from scipy.stats import entropy
from skimage import filters
from skimage import data, img_as_float
from scipy import ndimage
from scipy import stats
from scipy import stats as scipy_stats
from scipy.stats import skew, kurtosis
import threading

from preprocessing.extract_frames import process_directory as extract_from_avi
import preprocessing.grey_subset as preprocess_subsets

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.simplefilter("ignore", category=CryptographyDeprecationWarning)

# suppress specific RuntimeWarnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# path to the folder
def extract_frames_from_avi(avi_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    extract_from_avi(os.path.dirname(avi_file), "extracting frames", specific_file=avi_file)
    print(f"Extracted frames from {avi_file} to {output_folder}.")

def remap2(array_3d):
    # load the model from the file
    model_location = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models','colourmap', 'model.joblib')
    model = load(model_location)
    grayscale_scan = npy.zeros(array_3d.shape[:3], dtype=npy.uint8)
    for slice_idx in range(array_3d.shape[0]):
        image = array_3d[slice_idx, :, :]
        # reshape the image to a 2D array
        reshaped_image = image.reshape(-1, 3)
        # predict the class for each pixel
        predictions =  npy.max(npy.ceil(model.predict(reshaped_image)).astype(npy.uint8),axis=-1)
        
        greyscale = predictions.reshape(image.shape[:2])  # reshape back to 2D (grayscale)
        
        # store the grayscale image in the 3D array
        grayscale_scan[slice_idx, :, :] = greyscale
    return grayscale_scan


def process_folder(folder, force_redo=False):
    """
    Process a single folder through all steps, skipping if files exist unless force_redo is True.
    """
    # step 1: Check OCT_stack.npz (assumed to exist for folders)
    oct_stack_path = os.path.join(folder, 'OCT_stack.npz')
    if not os.path.exists(oct_stack_path):
        print(f"Warning: {oct_stack_path} missing. Assuming .avi extraction needed first.")
        return

    # step 2: Create grey_oct.npz
    grey_oct_path = os.path.join(folder, 'grey_oct.npz')
    if force_redo or not os.path.exists(grey_oct_path):
        stack3d = npy.load(oct_stack_path)
        oct_scan = stack3d['additional_crops']
        processed_slices = []
        for slice_y in range(oct_scan.shape[0]):
            slice_2d = oct_scan[slice_y, :400, :]
            is_greyscale = npy.all(slice_2d[:, :, 0] == slice_2d[:, :, 1], axis=0) & \
                           npy.all(slice_2d[:, :, 1] == slice_2d[:, :, 2], axis=0)
            greyscale_width = npy.argmax(~is_greyscale)
            if greyscale_width > 0:
                slice_2d = slice_2d[:, greyscale_width:, :]
            greyscale_slice = npy.sum(slice_2d, axis=-1)
            binary_slice = greyscale_slice > npy.percentile(greyscale_slice, 0.05)
            distance_transformed = ndi.distance_transform_edt(binary_slice)
            coordinates = npy.where(distance_transformed > 5)  # simplified from peak_local_max
            mask = npy.zeros_like(binary_slice, dtype=bool)
            mask[coordinates] = True
            mask_expanded = npy.repeat(mask[:, :, npy.newaxis], slice_2d.shape[2], axis=2)
            filtered_slice = npy.where(mask_expanded, slice_2d, 0)
            processed_slices.append(filtered_slice)

        max_width = max(slice.shape[1] for slice in processed_slices)
        filtered_scan = npy.zeros((oct_scan.shape[0], 400, max_width, 3), dtype=npy.uint8)
        for i, slice_2d in enumerate(processed_slices):
            filtered_scan[i, :slice_2d.shape[0], :slice_2d.shape[1], :] = slice_2d
        filtered_scan = remap2(filtered_scan)
        npy.savez_compressed(grey_oct_path, grey=filtered_scan)
        print(f"Saved {grey_oct_path}.")

