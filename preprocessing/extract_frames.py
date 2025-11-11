import os
import cv2
import numpy as npy
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.simplefilter("ignore", category=CryptographyDeprecationWarning)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    additional_crops = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # crop the frame to the 496x496 region from the top-left corner
        cropped_frame = frame[:496, :496]
        frames.append(cropped_frame)
        
        # crop the frame to the 496x??? region starting from (0, 496)
        additional_crop = frame[:496, 496:]
        additional_crops.append(additional_crop)
    
    cap.release()
    return frames, additional_crops

def save_cropped_frames_as_npz(frames, additional_crops, filename='cropped_frames.npz'):
    npy.savez_compressed(filename, frames=npy.array(frames), additional_crops=npy.array(additional_crops))


def save_green_pixel_positions(frames, filename='green_pixel_positions.npz'):
    all_greens = []
    for frame in frames:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = npy.array([40, 100, 100])
        upper_green = npy.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_mask = (mask == 255).astype(bool)
        all_greens.append(green_mask)

    green_frames = npy.array(all_greens)
    
    npy.savez_compressed(filename, green_positions=green_frames)

def remove_green_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = npy.array([40, 100, 100])
    upper_green = npy.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    frame_no_green = frame.copy()
    
    for j in range(frame.shape[0]):
        for i in range(frame.shape[1]):
            if mask[j, i] == 255:
                neighbors = frame_no_green[max(j-2, 0):min(j+3, frame_no_green.shape[0]), max(i-2, 0):min(i+3, frame_no_green.shape[1])]
                non_green_neighbors = neighbors[mask[max(j-2, 0):min(j+3, frame_no_green.shape[0]), max(i-2, 0):min(i+3, frame_no_green.shape[1])] == 0]
                
                if non_green_neighbors.size > 0:
                    frame_no_green[j, i] = npy.mean(non_green_neighbors, axis=0)
                else:
                    frame_no_green[j, i] = frame[j, i]
    
    return frame_no_green

def align_and_average_frames(frames):
    processed_frames = [remove_green_lines(frame) for frame in frames]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in processed_frames]
    ref_frame = gray_frames[0]
    aligned_frames = [ref_frame]
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_frame, None)
    
    for frame in gray_frames[1:]:
        kp2, des2 = orb.detectAndCompute(frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        points1 = npy.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = npy.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if len(points1) >= 3 and len(points2) >= 3:
            warp_matrix, _ = cv2.estimateAffinePartial2D(points2, points1)
            aligned_frame = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]))
            aligned_frames.append(aligned_frame)
    
    average_frame = npy.mean(aligned_frames, axis=0).astype(npy.uint8)
    return average_frame

def main(avi_path):
    os.path.dirname(avi_path)
    res_dir = os.path.join(os.path.dirname(avi_path), os.path.basename(avi_path)[:-4])
    os.makedirs(res_dir, exist_ok=True)
    print('loading avi file...')
    video_path = avi_path
    print('extracting frames...')
    frames, additional_crops = extract_frames(video_path)
    print('saving cropped frames...')
    save_cropped_frames_as_npz(frames, additional_crops, os.path.join(res_dir, 'OCT_stack.npz'))
    
    # save green pixel positions from the first frame
    print('saving green pixel positions...')
    save_green_pixel_positions(frames, os.path.join(res_dir, 'green_frame.npz'))
    
    frames = [remove_green_lines(frame) for frame in frames]
    average_frame = align_and_average_frames(frames)
    
    cv2.imwrite(os.path.join(res_dir,'averaged_frame.png'), average_frame)
    
def process_file(avipath):
    try:
        main(avipath)
    except Exception as e:
        print(f"Error processing {avipath}: {str(e)}")

def process_directory(input_path, desc='directories', specific_file=None):
    if specific_file:
        avi_files = [specific_file] if os.path.isfile(specific_file) and specific_file.endswith('.avi') else []
    else:
        avi_files = [os.path.abspath(os.path.join(input_path, item)) 
                     for item in os.listdir(input_path) 
                     if os.path.isfile(os.path.join(input_path, item)) and item.endswith('.avi')]
    
    num_processes = min(cpu_count(), 32)
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_file, avi_files), total=len(avi_files), desc=desc))

