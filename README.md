# OCTOPUS (Optical Coherence Tomography Ocular Pathology by Unsupervised Segmentation)
An application for retinal OCT image analysis.

## 1. Introduction
This manual provides documentation for the OCTOPUS application, a tool designed for processing and analyzing optical coherence tomography (OCT) images of retinal structures. The application supports tasks such as data preprocessing, model-based prediction of pathological features, manual annotation, visualization of fundus images, statistical computation of annotated areas, and timeline-based tracking of changes over multiple timepoints. It is intended for researchers and practitioners in biomedical imaging who require structured analysis of retinal OCT datasets.

The application operates on image stacks derived from AVI files (Spectralis output format), using a deep learning model to identify regions of interest. All operations are performed within a graphical user interface (GUI) built using PyQt5, with data persistence handled through NumPy arrays and JSON metadata.

## 2. System Requirements and Installation
### 2.1 System Requirements
- Hardware: Minimum 8 GB RAM recommended.
- Operating System: Cross-platform (Python, tested on Windows).
- Python Version: 3.12 or later.
- Dependencies: 

| Package Name	 | Installation Command |
|----------------|----------------------|
| cryptography | 	pip install cryptography |
| cv2 | 	pip install opencv-python |
| joblib | 	pip install joblib |
| matplotlib | 	pip install matplotlib |
| numpy | 	pip install numpy |
| onnxruntime | 	pip install onnxruntime |
| pandas | 	pip install pandas |
| PyQt5 | 	pip install PyQt5 |
| qdarkstyle | 	pip install qdarkstyle |
| scikit-image | 	pip install scikit-image |
| scikit-learn | 	pip install scikit-learn |
| scipy | 	pip install scipy |
| tqdm | 	pip install tqdm |



### 2.2 Installation
Ensure the Dependencies listed above are installed in the environment to be used.

A snapshot of the model weights is available at https://gitlab.fel.cvut.cz/mikroapo/octopus-model-weights or https://doi.org/10.5281/zenodo.17580406, under the terms of the CC-BY 4.0 license.
Copy the UNet.onnx file to the models/UNet directory.

\* The app can be used without a model, and will still do preprocessing, viewing, manual annotations, and exporting, but requests to run model predictions will fail.

## 3. Getting Started
### 3.1 Launching the Application
Run the application by executing `python octopus.py` in a terminal. The main window will appear with dimensions approximately 1600x960 pixels, adjustable based on screen resolution.

### 3.2 Initial Configuration
Upon launch, you can change the log directory via the "log location" button to enable session logging. This directory will store session logs, notes, and exported statistics.

By default, the app will create a "logs" subfolder under the directory selected in the "open" dialog box.

## 4. Main Interface Overview
The GUI is divided into several group boxes for organization:
### 4.1 Data Loading and Preprocessing Group
- Buttons: 
> - Open folder dialog<br/>
> - Run preprocess script<br/>
> - Load stacks from selected folders<br/>
> - Batch process all (all processed folders in the current directory - no listbox selection nededed)<br/>
- Checkbox: Force redo for the preprocessing.
- List Widget: Displays selected folders or AVI files.
- Progress Bar: Shows preprocessing progress.

### 4.2 Model Configuration Group
- Button: Open U-Net dialog. This can be skipped if the desired model is already in the models/UNet directory - it will be selected automatically when requesting the first model run.
- Label: Displays loaded model location.
- SpinBox: Threshold for probability-based annotation (default: 0.5).
- Radio Buttons: Scoring method - max probability or 99th percentile (to avoid outliers, e.g. fewer than 3 pixels with predicted probability of 1).

### 4.3 Slice View Group
- Graphics View: Displays the current OCT slice with optional overlays for annotations and probabilities.

### 4.4 Fundus View Group
- Graphics View: Shows the averaged fundus image with overlays for slice positions, bounding boxes, and projected annotations.
- Double-click in the fundus image to hide/show the overlays. Ensure overlays are shown when exporting the stack, to include them in the SVG.

### 4.5 Navigation and Editing Group
#### Stack box (left)
- Buttons: Previous/Next stack, Run model on stack, Interrupt batch processing (this works for interrupting all batch processes - entire stack & multiple stacks in the folder).

#### Slice box
- Slider: Move between slices within the stack.
- Buttons: Previous/Next stack, Previous/Next slice, Run model on stack/single slice, Interrupt.
- Checkbox: Enable highlighter for manual editing.
- Buttons: Add/Clear annotation, Clear all annotations.

### 4.6 Display and Settings Group
- Checkbox: Show model probability score.
- SpinBox: Fundus scale bar length in μm.

#### timeline function
- Button: Timeline (opens separate window).

### 4.7 Statistics and Export Group
#### stats window
- Displays: Total annotated area (μm²), number of regions.

#### export box
- Checkbox: Overwrite saved files.
- Text Box: Log note for current stack.
- Buttons: Append note, Export stack/slice (SVG), Export statistics (CSV).

## 5. Preprocessing Data
To allow the app to recognise different animals, left/right eyes and timepoints, save the .avi filenames in this format:<br/>
>mouse-ID_L_YYYY-MM-DD (L for left eye)<br/>
>mouse-ID_R_YYYY-MM-DD (R for right eye)<br/>
Avoid underscores ( _ ) in the mouse-ID, to prevent confusion

For example:<br/>
<b>example-1-ABC0001_L_2025-01-31</b> will match the left eye from subject with ID "example-1-ABC0001", examined on 2025-01-31 (January 31st, 2025)

### 5.1 Extracting Frames from AVI Files (Batch Preprocessing)
Open a filder ("open" button) and select AVI files (hold CTRL to select multiple). 

Enable "force redo" to overwrite existing files. 

Use "preprocess" to extract the components from video frames. Monitor progress via the bar.

This will generate folders (with the same name as the AVI file) that can be used by the app for display/annotations/predictions.

## 6. Loading and Navigating Image Stacks
### 6.1 Loading Stacks
Select preprocessed folders (hold CTRL or SHIFT to select multiple) and use "load" Stacks are loaded into memory as 3D NumPy arrays.

### 6.2 Navigation
Use keyboard shortcuts (e.g., A/D for slices, Shift+A/D for stacks) or buttons to switch between stacks and slices. The slice view updates automatically.

## 7. Model-Based Prediction
### 7.1 Loading Models
Load U-Net models via the dialog. Models predict probabilities for pathological regions. You can skip this step if the desired model is in the models/UNet directory already.

### 7.2 Running Predictions
- Single Slice: Use "Run model (1x)" or shortcut R.
- Entire Stack: Use "Run model" or Shift+R; clicking the "interrupt" button will stop the prediction at the end of the current slice.


- Batch: "run all" process will auto-load all the pre-processed folders it can find, run the prediction for the entire stack for each one, and export to CSV for all.

Predictions update probability arrays, which can be thresholded for annotations.

## 8. Manual Annotation
### 8.1 Enabling Editing
Check "edit (highlight)" (shortcut E) to allow mouse-based selection in the slice view.

### 8.2 Adding and Removing Annotations
Drag on the slice image to highlight regions; use "Add" (W) or "Clear" (S) buttons to manually add or remove annotations from the highligted region. Clear all with Shift+C.

Note that running the model (Shortcut: R) will redo the annotation based on the model predictions, and thus over-write any manual annotations.

Annotations are stored as 1D arrays per slice and projected onto the fundus view. They are saved when they are created / updated, so there is no traditional "save" option - only export what you need.

## 9. Fundus View and Overlays
### 9.1 Displaying Fundus Images
Loaded automatically from averaged_frame.png when loading a stack. Overlays include green bounding boxes, slice position lines, the currently displayed slice as a yellow line, and blue annotated areas.

### 9.2 Interactivity
Double-click to toggle overlays. Annotations are projected and connected across slices for area computation.

## 10. Statistics and Export
### 10.1 Computing Statistics
Areas are calculated in (fundus image) pixels and converted to μm² using the scale bar. Displays total area and region count.
The area between successive slices is calculated by linear interpolation.

### 10.2 Exporting Data
- SVG: Export slice or stack views, with optional timestamp (leave "overwrite" unchecked to get unique timestamps).
- CSV: Statistics appended to statistics.csv in the log directory.

## 11. Timeline View
### 11.1 Accessing the Timeline
Click "Timeline" to open a separate window.

### 11.2 Generating Timelines
Select an animal ID and eye from the combo box, then "Generate". Displays thumbnails, statistics, and a plot of annotated areas over time.

This will use all the loaded folders it can find that match the criteria (same ID, same eye, multiple dates).

### 11.3 Interactivity
Click plot points to jump to stacks.

## 12. Logging and Notes
### 12.1 Session Logging
Actions are logged to a <i>session.log</i> file in the logs/outputs directory.

You can add stack-specific notes via the text box and "append" button.

### 12.2 Data Persistence
Annotations and probabilities are saved to .npy files with metadata for integrity checks.

## 13. Troubleshooting and Known Limitations
### 13.1 Common Issues
- Missing Files: Ensure grey_oct.npz exists post-preprocessing.
- Model Errors: Verify ONNX model compatibility.
- Performance: Large stacks need more memory.

### 13.2 Limitations
- Designed for Spectralis AVI output with jet colourmap.
- Assumes specific avi file naming convention.

## Appendix: Keyboard Shortcuts
>E : enable highlighter for annotation editing<br/>
>Shift+C : clear annotaions from current slice<br/>
>R : run the model for the current slice<br/>
>Shift+R : run the model for the entire stack<br/>
>WASD cluster:
>>A-D : previous-next slice<br/>
>>W-S : manually add-remove annotation from the highlighted area<br/>
>> Shift + A-D : previous-next stack<br/>


## Licensing and Copyright notice
This software is released under the terms of the GNU General Public License v3.0 (GPL-3.0). You can redistribute it and/or modify it under the terms of the GPL-3.0 as published by the Free Software Foundation.

© Apostolos Mikroulis, Department of Cybernetics, Faculty of Electrical Engineering, Czech Technical University, 2025.

### Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.