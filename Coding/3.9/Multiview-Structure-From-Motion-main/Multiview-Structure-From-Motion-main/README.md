# Structure-from-Motion (SfM)

![SfM- banner](Results/GustavllAdolf.png)

## Overview

This repository contains a Structure from Motion(SfM) pipeline that resonctructs 3D point clouds from a set of continuous images of a scene. The project aims to demonstrate the fundamentals of camera calibration, feature matching, feature matching, pose estimation, triangulation of 3D points and bundle adjustment(optional) to refine both 3D pointds and camera parameters and point cloud visualization which is a 3D representation of the captured scene.

**Key Highlights:**
- **Incremental SfM:** Takes pairs of images, estimates relative pose, and gradually adds more images to improve the reconstruction.
- **Feature Matching:** Uses SIFT + FLANN-based matching with ratio test for robust keypoint correspondences.
- **3D Reconstruction:** Triangulates matched points across multiple views to produce a 3D point cloud.
- **Bundle Adjustment (optional):** Refinement of camera poses and 3D structure via iterative least squares.
- **PLY Output:** Saves the final 3D point cloud (with color) in .ply format for visualization in tools like Open3D or MeshLab.

---

## Features

- **Camera Intrinsic Handling**  
  - Loads a custom `K.txt` file containing the 3×3 camera intrinsic matrix. 
  - Automatically scales intrinsics if downsampling the input images

- **Robust Feature Matching**  
  - SIFT descriptors (up to 10,000 features if needed).  
  - FLANN-based nearest neighbor search with ratio test to remove ambiguous matches(outliers).

- **Pose Estimation**  
  - Essential matrix estimation + `recoverPose` for the initial two images 
  - `solvePnPRansac` for subsequent images to localize new cameras.

- **Triangulation**  
  - Converts matched points into 3D coordinates with `cv2.triangulatePoints`.

- **Bundle Adjustment(Optional)**  
  - Minimizes reprojection error across cameras and 3D points for a more accurate reconstruction if enabled.

- **PLY Export**  
  - Saves final 3D points + color in ASCII or binary PLY format.  
  - Includes optional outlier removal and user-defined scaling factor.

---

## Methodology

The **Structure from Motion(SfM)** pipeline reconstructs a 3D representation of an object from a continuous series of continuos 2D images of an object from different angles by estimating camera poses and triangulating the matched feature points. Here is the stp-by-step implementation of the pipeline:

1. **Image & Intrinsics Loading**  
   - Read all images from a dataset directory capturing the object from different viewpoints.  
   - Read `K.txt` for the 3×3 camera intrinsics.  
   - Downsamples images and scales the intrinsic matrix accordingly.

2. **Feature Detection and Matching**  
   - **Feature Detection:** Use SIFT to detect and compute descriptors for the iamges. 
   - **Feature Matching:** Utilize a FLANN-based matcher to finds matches between consecutive images.  
   - **Lowe's Ratio Test:** filters ambiguous(outliers) matches.

3. **Initial Pose Estimation**  
   - **Essential Matrix estimation:** Compute essential matrix using th matched feature points and recover pose for the first two images.  
   - **Pose Recovery:** Recover fundamental camera motion (rotation, translation) under the pinhole camera model.

4. **Triangulation:**
    - **Project Matrices:** Construct projection matrices for each camera pose.
    - **3D points Triangulation:** Convert 2D matched points into 3D coordinates. Maintain a growing set of 3D points as more images are added.

5. **Incremental SfM**  
   - For each new image:
     - Match features to the previous image.  
     - `SolvePnPRansac` to get the new camera pose.  
     - Triangulate points between the new view and an existing view. 

6. **Bundle Adjustment(Optional)** 
   - If enabled, refine all camera parameters and 3D points by minimizing reprojection error.

7. **Point Cloud Export**  
   - Accumulate all 3D points plus colors from each iteration.  
   - Save to PLY with potential outlier removal, scaling, and color normalization.

---

## Installation

### Prerequisites
**Python 3.7+**
  ```bash
  Install Additional Dependencies listed in the 'Requirements.txt' file.
  ```

### Setup Instructions

1. **Clone the Repository:**

    ```bash 
    git clone https://github.com/StarkGoku10/Multiview-Structure-From-Motion.git
    cd Multiview-Structure-From-Motion
    ```

2. **Create a Virtual Environment(Optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate #on windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash 
    pip install -r Requirements.txt
    ```

4. **Visualization(Optional)**

    For visualizing the saved point cloud, python's `Open3D` library is used. 
    ```bash
    pip install open3d
    ```

5. **Important Note:**
  - **Images:** 
      - Ensure there is sufficient overlap of the object/scene between consecutive images.
      - Capture images with different orientations and perspectives of the scene.

  - **Calibration File**(`K.txt`)**:**
      - This file contains the camera intrinsic parameters(matrix). The format expected for this file is nine numerical values in a single line or representating 3x3 matrix seperated by spaces or new lines.
      - Ensure that the matrix is accurate and corresponds to the camera used to capture the images.
      - **Example `K.txt` matrix:**

          ```bash 
          2759.48 0 1520.69
          0 2764.16 1006.81
          0 0 1
          ```

---

## Usage

### Executing the SfM pipeline

Execute the `sfm.py` script to run the SfM pipeline. The script processes the first two images and triangulates the points between the two images. Then the pipeline adds one image sequentially to reconstruct the D structure of the scene. Optionally, **Bundle Adjustment** can be enabled to refine the 3D points and reduce the reprojection errors.

**Steps:**

1. **Navigate to the Project Directory**

    Ensure you are in the root directory where the 'SfM.py` file resides.

    ```bash
    cd Multiview-Structure-from_Motion
    ```

2. **Run the Pipeline**

    **Note:** Adjust the dataset path as needed(in the `__main__` block):

    ```bash
    if __name__ == '__main__':
      sfm= StructurefromMotion("Datasets/YourDataset")
      sfm()
    ```
    Ensure you have specified the proper directory paths for the dataset. Execute the following command: 

    ```bash
    python SfM.py
    ```

3. **Monitor the Output:**

  - The script will process first two first and display messages about the processing step. 
  - For each subsequent step, it will perform featue matching,pose estimation, pose recovery, trianglation, bundle adjustment(if enabled) and display the iamge and reprojection error graph for all the images.
  - The script will save the final 3D reconstruction in the form of a point cloud(`.ply`) and the pose array in the the respective folders. 

4. **Visualize the Point Cloud:**
    
   - The point cloud can be visualized using visualized tools like [MeshLab](https://www.meshlab.net/), [Open3D](https://www.open3d.org/l) or [CloudCompare](https://www.cloudcompare.org/).
   - For this project, `Open3D` is used for visualzation of the point cloud. To install `open3d`, Execute the following command:
   
      ```bash
      pip install open3d 
      ```

   - Run the script with the appropriate path to the point cloud to generate the 3D visualization:

      ```bash
      python visualize.py
      ``` 

---

## Output 

1. **Console:**

    ```bash
      Camera Intrinsic Matrix: [[ 1.1969761e+03 -3.4106051e-13  4.6619110e+02]
      [ 0.0000000e+00  1.1990593e+03  3.1413251e+02]
      [ 0.0000000e+00  0.0000000e+00  1.0000000e+00]]
      Reprojection error for first two images: 0.00487721240747486
      total_images 55
        0%|                                                                                                                                                                             | 0/55 [00:00<?, ?it/s] Shape of New Array (468, 2) (468, 2)
      Reprojection error: 0.0508255796928927
        2%|███                                                                                                                                                                  | 1/55 [00:01<00:58,  1.08s/it] Shape of New Array (295, 2) (295, 2)
      Reprojection error: 0.5636247818285103
        4%|██████                                                                                                                                                               | 2/55 [00:01<00:37,  1.40it/s]
        ...
    ```

2. **Reprojection Error Plot:**

    - Saved in `/Results` directory.

3. **Point Cloud(`.ply`):**

    - Point cloud will be saved in directory `/Datasets/<DatasetName.ply>`.
    - Contains the 3D points plus per-point color(in ASCII or binary).

4. **Pose Array:**

    - A CSV file genereated will stored in `/Results Array/<DatasetName>_pose_array.csv` with all the camera parameters.

---

## Results 

### **Results(without Bundle Adjustment)**:

1. **Herz-Jesus:**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Herz-Jesus-P8/reprojection_errors_herz.png" alt="Image 1" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Herz-Jesus-P8/herz-jesues.gif" alt="GIF 1" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

2. **Fountain-P11**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Fountain-P11/reprojection_errors_fountain.png" alt="Image 2" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Fountain-P11/fountain-P11.gif" alt="GIF 2" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

3. **Gustavll-Adolf**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results without Bundle Adjustment/GustavllAdolf/reprojection_errors_Gustav.png" alt="Image 3" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results without Bundle Adjustment/GustavllAdolf/GustavllAdolf.gif" alt="GIF 3" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

4. **Entry-P10**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Entry-P10/reprojection_errors_entry.png" alt="Image 3" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results without Bundle Adjustment/Entry-P10/Entry.gif" alt="GIF 3" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

### **Results(with Bundle Adjustment)**:

1. **Herz-Jesus:**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results with Bundle Adjustment/Herz-Jesus/reprojection_errors_herz_ba.png" alt="Image 1" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results with Bundle Adjustment/Herz-Jesus/herz.gif" alt="GIF 1" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

2. **Fountain-P11**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results with Bundle Adjustment/Fountain-P11/reprojection_errors_fountain_ba.png" alt="Image 2" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
    <td>
      <figure style= "text-align: center;">
        <img src="Results/Results with Bundle Adjustment/Fountain-P11/fountain.gif" alt="GIF 2" width="300">
        <figcaption>Reconstruction</figcaption>
      </figure>
    </td>
  </tr>
</table>

3. **Entry-P10**

<table>
  <tr>
    <td>
      <figure style="text-align: center;">
        <img src="Results/Results with Bundle Adjustment/Entry-P10/reprojection_errors_entry_ba.png" alt="Image 3" width="300">
        <figcaption>Reprojection Error plot</figcaption>
      </figure>
    </td>
  </tr>
</table>
