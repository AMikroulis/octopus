import os
import numpy as npy
import json
from datetime import datetime
import zlib


class PersistenceManager:
    @staticmethod
    def save_arrays(stack_path, stack_index, probabilities_array, annotations_array):
        try:
            prob_path = os.path.join(stack_path, 'probabilities.npy')
            anno_path = os.path.join(stack_path, 'annotations.npy')
            meta_path = os.path.join(stack_path, 'metadata.json')
            
            # save the arrays to disk
            npy.save(prob_path, probabilities_array)
            npy.save(anno_path, annotations_array)
            
            # calculate CRC32 hashes for each array
            prob_hash = zlib.crc32(probabilities_array.tobytes())
            anno_hash = zlib.crc32(annotations_array.tobytes())
            
            # create metadata with hashes
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'folder_name': os.path.basename(stack_path),
                'prob_hash': prob_hash,
                'anno_hash': anno_hash
            }
            
            # save metadata to JSON file
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
                
            print(f"Saved arrays for stack {stack_index} at {stack_path}")
        except Exception as e:
            print(f"Error saving arrays for stack {stack_index}: {e}")
    
    @staticmethod
    def load_arrays(stack_path, stack_index, num_slices, slice_width):
        prob_path = os.path.join(stack_path, 'probabilities.npy')
        anno_path = os.path.join(stack_path, 'annotations.npy')
        meta_path = os.path.join(stack_path, 'metadata.json')
        
        try:
            # load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # load arrays
            probabilities_array = npy.load(prob_path)
            annotations_array = npy.load(anno_path)
            
            # verify hashes
            prob_hash = zlib.crc32(probabilities_array.tobytes())
            anno_hash = zlib.crc32(annotations_array.tobytes())
            
            if prob_hash != metadata['prob_hash'] or anno_hash != metadata['anno_hash']:
                print(f"Hash mismatch for stack {stack_index} - data may be corrupted!")
                # return empty arrays as a fallback
                return npy.zeros((num_slices, slice_width), dtype='float32'), npy.zeros((num_slices, slice_width), dtype='uint8')
            
            print(f"Loaded arrays for stack {stack_index}")
            return probabilities_array, annotations_array
        except Exception as e:
            print(f"Error loading arrays for stack {stack_index}: {e}")
            return npy.zeros((num_slices, slice_width), dtype='float32'), npy.zeros((num_slices, slice_width), dtype='uint8')
    
    @staticmethod
    def is_data_valid(stack_path, stack_index, stack_file_path):
        """Check if saved data is up-to-date based on timestamp."""
        meta_path = os.path.join(stack_path, 'metadata.json')
        stack_file = stack_file_path  # e.g., path to grey_oct.npz
        
        if not os.path.exists(meta_path) or not os.path.exists(stack_file):
            return False
            
        try:
            # load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            saved_time = datetime.fromisoformat(metadata['timestamp'])
            
            # get stack file modification time
            stack_mtime = datetime.fromtimestamp(os.path.getmtime(stack_file))
            
            # data is valid if saved after the stack file was last modified
            return saved_time >= stack_mtime
        except Exception as e:
            print(f"Error checking timestamp for stack {stack_index}: {e}")
            return False
    
    @staticmethod
    def save_settings(stack_path, stack_index, threshold, scalebar_um):
        """Save spinbox settings (threshold and scalebar) to a JSON file."""
        try:
            settings_path = os.path.join(stack_path, 'settings.json')
            settings = {
                'threshold': threshold,
                'scalebar_um': scalebar_um,
                'timestamp': datetime.now().isoformat()
            }
            with open(settings_path, 'w') as f:
                json.dump(settings, f)
            print(f"Saved settings for stack {stack_index}")
        except Exception as e:
            print(f"Error saving settings for stack {stack_index}: {e}")
    
    @staticmethod
    def load_settings(stack_path, stack_index):
        """Load spinbox settings, returning None if not found."""
        settings_path = os.path.join(stack_path, 'settings.json')
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                return settings.get('threshold'), settings.get('scalebar_um')
            return None, None
        except Exception as e:
            print(f"Error loading settings for stack {stack_index}: {e}")
            return None, None