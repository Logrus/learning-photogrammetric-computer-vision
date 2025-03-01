#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



def plot_matches(image1: np.ndarray, image2: np.ndarray, 
                 coords1: np.ndarray, coords2: np.ndarray, 
                 matches: np.ndarray) -> None:
    """
    Visualizes matching keypoints between two grayscale images by concatenating them 
    side by side and drawing lines connecting matching points.

    This function creates a composite image from two input images by placing them 
    next to each other. It then plots the keypoints from both images (using red crosses) 
    and draws a yellow line between each pair of matched keypoints.

    Parameters:
        image1 (np.ndarray): First grayscale image as a 2D array of shape (height, width).
        image2 (np.ndarray): Second grayscale image as a 2D array of shape (height, width).
        coords1 (np.ndarray): Keypoint coordinates in the first image, with shape (N1, 2) 
                              where each row represents (x, y) coordinates.
        coords2 (np.ndarray): Keypoint coordinates in the second image, with shape (N2, 2) 
                              where each row represents (x, y) coordinates.
        matches (np.ndarray): Array of matched keypoint indices with shape (M, 2), where each 
                              row is (index_in_coords1, index_in_coords2).

    Returns:
        None. The function displays the resulting plot.
    """
    # Determine the dimensions of the composite image.
    width_combined = image1.shape[1] + image2.shape[1]
    height_combined = max(image1.shape[0], image2.shape[0])
    
    # Create an empty composite image.
    composite_image = np.zeros((height_combined, width_combined), dtype=np.uint8)
    
    # Place the first image on the left.
    composite_image[:image1.shape[0], :image1.shape[1]] = image1
    # Place the second image on the right.
    composite_image[:image2.shape[0], image1.shape[1]:] = image2

    # Set up the plot.
    plt.figure(figsize=(10, 8))
    plt.imshow(composite_image, cmap="gray")
    plt.axis("off")
    
    # Loop through each match and plot the keypoints and connecting lines.
    for match in matches:
        idx1, idx2 = match
        point1 = coords1[idx1]
        # Offset the x-coordinate of the second image keypoints.
        point2 = coords2[idx2] + np.array([image1.shape[1], 0])
        
        # Plot keypoints as red crosses.
        plt.plot(point1[0], point1[1], "rx", markersize=5)
        plt.plot(point2[0], point2[1], "rx", markersize=5)
        
        # Draw a yellow line connecting the keypoints.
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], "-y", linewidth=1)
    
    # Display the final plot.
    plt.show()

def plot_matches2(I1, I2, C1, C2, M):
    """
    Plots the matches between the two images
    """
    # Create a new image with containing both images
    W = I1.shape[1] + I2.shape[1]
    H = np.max([I1.shape[0], I2.shape[0]])
    I_new = np.zeros((H, W), dtype=np.uint8)
    I_new[0 : I1.shape[0], 0 : I1.shape[1]] = I1
    I_new[0 : I2.shape[0], I1.shape[1] : I1.shape[1] + I2.shape[1]] = I2

    # plot matches
    plt.imshow(I_new, cmap="gray")
    for i in range(M.shape[0]):
        p1 = C1[M[i, 0], :]
        p2 = C2[M[i, 1], :] + np.array([I1.shape[1], 0])
        plt.plot(p1[0], p1[1], "rx")
        plt.plot(p2[0], p2[1], "rx")
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-y")