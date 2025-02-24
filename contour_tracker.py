import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv


def fill_holes_before_binarization(image):
    """
    Fill holes in the image before binarization (regions of black pixels inside white clusters).
    """
    kernel = np.ones((3, 3), np.uint8)  # Kernel size can be adjusted
    filled_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return filled_image


def binarize_by_color(image, lower_hsv, upper_hsv):
    """
    Binarize the image based on a specific color range in HSV.
    - lower_hsv, upper_hsv: Tuple of (H, S, V) defining the color range.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask


def fill_holes_after_binarization(image):
    """
    Fill holes in the binarized image (regions of black pixels inside white clusters).
    """
    kernel = np.ones(
        (3, 3), np.uint8)  # Kernel size can be adjusted (larger kernel)
    filled_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return filled_image


def remove_small_clusters(image, min_percentage):
    """
    Find connected components and remove clusters smaller than a given percentage of the total image size.
    """
    # Get the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the minimum number of pixels for a cluster based on the percentage
    min_area = int(total_pixels * min_percentage / 100)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for large enough clusters
    mask = np.zeros_like(image)

    for contour in contours:
        # Check the area of each contour
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to keep only clusters that are larger than the min_area
    result = cv2.bitwise_and(image, mask)

    return result


def remove_all_but_largest_black_cluster(image):
    """
    Finds the largest black cluster in a binary image and turns all other black clusters white.
    """
    # Invert the image so black regions become white (for contour detection)
    inverted = cv2.bitwise_not(image)

    # Find contours of black clusters
    contours, _ = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # No clusters found, return original image

    # Find the largest black cluster based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a new white image
    result = np.full(image.shape, 255, dtype=np.uint8)

    # Draw only the largest black cluster in black
    cv2.drawContours(result, [largest_contour], -1, 0, thickness=cv2.FILLED)

    return result


def crop_to_extreme_clusters(image, contours):
    """
    Crop the image to include all clusters, determining the edges based on the extreme positions 
    of all the clusters (left, right, top, bottom).
    """
    # Initialize extreme values
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Find the extreme values for all clusters
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y)
        max_y = max(max_y, y + h)

    # Crop the image to the bounding box that includes all clusters
    cropped_image = image[min_y:max_y, min_x:max_x]

    return cropped_image


def connect_bottom_of_clusters(image, contours):
    """
    Connect the bottom of the clusters with a white line and set everything below this line to white.
    """
    # Get the height and width of the image
    height, width = image.shape

    # Make a copy of the image to manipulate
    result_image = image.copy()

    # Find the bottommost point for each cluster
    bottom_points = []

    # Set all pixels below the white line to white
    for x, y in bottom_points:
        # Set all pixels below each bottom point white
        result_image[y:, x] = 255

    for contour in contours:
        # Get the y-coordinate of the bottommost point of the contour (maximum y-value)
        bottom_y = max(contour[:, 0, 1])
        # Get the x-coordinates of all the points in the contour
        x_coords = contour[:, 0, 0]
        # Find the corresponding x-coordinate at the bottom of the contour
        bottom_x = x_coords[np.argmax(contour[:, 0, 1])]
        bottom_points.append((bottom_x, bottom_y))

    # Sort bottom points by their x-coordinate
    bottom_points.sort(key=lambda x: x[0])

    # Set all pixels below the white line to white
    for x, y in bottom_points:
        # Set all pixels below each bottom point white
        result_image[y:, x] = 255

    # Now fill the whole area below the connected line with white pixels
    for i in range(len(bottom_points) - 1):
        x1, y1 = bottom_points[i]
        x2, y2 = bottom_points[i + 1]

        result_image[y1:, :] = 255  # Set all pixels below the line white

    return result_image


def find_topmost_point_at_each_x(image):
    """
    Find the topmost point for each x-coordinate in the image and return the y-values.
    """
    # Check if the image is already grayscale (binary)
    if len(image.shape) == 2:  # This means the image is already grayscale (1 channel)
        binary_image = image
    else:
        # Convert the image to grayscale if it's not already
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Initialize a list to store the topmost y-values for each x-coordinate
    topmost_y_values = []

    # Iterate over each x-coordinate of the image
    for x in range(binary_image.shape[1]):
        # Extract the column of pixels for the current x-coordinate
        column = binary_image[:, x]

        # Find the topmost white pixel (y-coordinate) in this column
        y_coordinates = np.where(column == 255)[0]

        if len(y_coordinates) > 0:
            # Get the smallest y-coordinate (topmost)
            topmost_y = np.min(y_coordinates)
        else:
            # If no white pixels, consider it as bottom of the image
            topmost_y = binary_image.shape[0]

        topmost_y_values.append(topmost_y)

    return topmost_y_values


def convert_pixels_to_mm(pixel_distances, scaling_factor):
    """
    Convert a list of pixel distances to millimeters using a scaling factor.
    """
    return [distance * scaling_factor for distance in pixel_distances]


def plot_values(topmost_y_values, scaling_factor):
    """
    Plot the topmost y-values at each x-coordinate and the corresponding distances in mm.
    """
    # Convert distances from pixels to millimeters
    topmost_y_values_mm = convert_pixels_to_mm(
        topmost_y_values, scaling_factor)

    # Create an x-coordinate array for the image width
    x_values = np.arange(len(topmost_y_values_mm))

    return [x_values * scaling_factor, topmost_y_values_mm]


def test_find_topmost_distance(image, scaling_factor):
    """
    Load the image, find the topmost point for each x-coordinate, and plot the distances in mm.
    """

    if image is None:
        print("Error: Could not read image from", image_path)
        return

    # Find the topmost y-values for each x-coordinate in the image
    topmost_y_values = find_topmost_point_at_each_x(image)

    # Plot the distances in mm
    return plot_values(topmost_y_values, scaling_factor)


def calculate_scaling_factor(binarized_image, ratio):
    """
    Detect the white scale rectangle in the bottom right of the image 
    and compute the scaling factor based on the longest side.
    Assumes that the longest side corresponds to 5mm in physical length.

    Parameters:
      - binarized_image: The processed image after binarization and hole filling.

    Returns:
      - scaling_factor: The number of pixels per mm calculated from the scale rectangle.
                        Returns None if the scale rectangle is not detected.
    """
    # Define the constant ROI: lower 5% of the y-axis and rightmost 10% of the x-axis

    # Convert to grayscale
    gray_image = cv2.cvtColor(binarized_image, cv2.COLOR_BGR2GRAY)

    # Apply binarization with a threshold of 0.5 (127 is half of 255, used for binarization)
    _, binarized_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    height, width = binarized_image.shape
    y_start = int(0.5 * height)
    x_start = int(0.7 * width)
    # Crop the image to the defined ROI
    roi_image = binarized_image[y_start:, x_start:]

    # Find contours in the cropped ROI
    contours, _ = cv2.findContours(
        roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_contour = None
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Ensure the polygon has 4 sides (i.e. is rectangular)
        if len(approx) == 4:
            # Calculate the bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            print(f"Rectangle detected at: (x={
                  x+x_start}, y={y+y_start}), width={w}, height={h}")
            # Filter out too-small rectangles (adjust threshold as needed)
            if w * h < 1000:
                continue
            scale_contour = approx
            break

    scaling_factor = None
    if scale_contour is not None:
        x, y, w, h = cv2.boundingRect(scale_contour)
        longest_side_pixels = max(w, h)
        # 5mm is the physical length of the longest side
        scaling_factor = ratio / longest_side_pixels
        cv2.rectangle(binarized_image, (x+x_start, y+y_start),
                      (x+x_start + w, y+y_start + h), (0, 255, 0), 100)
        print(f"Scale rectangle detected: Longest side = {
              longest_side_pixels} pixels")
        print(f"Scaling factor: {scaling_factor} mm per pixel")
    else:
        print("Scale rectangle not detected in the bottom-right ROI.")

    return scaling_factor


def add_units_to_ticks(ax, axis='x', unit='mm'):
    """
    Add units to the second-to-last tick on the given axis.

    Parameters:
      - ax: The axes object.
      - axis: 'x' or 'y', the axis to modify.
      - unit: The unit to append to the tick label (default 'mm').
    """
    if axis == 'x':
        old_xticks = [t for t in ax.get_xticks() if t >= ax.get_xlim()[
            0] and t <= ax.get_xlim()[1]]
        ax.set_xticks(old_xticks)
        old_xticklabels = ax.get_xticklabels()
        old_xticklabels[-2] = "mm"
        ax.set_xticklabels(old_xticklabels)

    elif axis == 'y':
        old_yticks = [t for t in ax.get_yticks() if t >= ax.get_ylim()[
            0] and t <= ax.get_ylim()[1]]
        ax.set_yticks(old_yticks)
        old_yticklabels = ax.get_yticklabels()
        old_yticklabels[-2] = "mm"
        ax.set_yticklabels(old_yticklabels)


def export_to_csv(xvals, yvals, filename="depth_data.csv"):
    """
    Export xvals and yvals to a CSV file.

    Parameters:
      - xvals: List of x coordinates (e.g., pixel positions).
      - yvals: List of y coordinates (e.g., welding depths).
      - filename: The name of the file to export (default 'depth_data.csv').
    """
    # Open a new CSV file to write the data
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Length [mm]", "Keyhole depth [mm]"])

        # Write data rows
        for x, y in zip(xvals, yvals):
            writer.writerow([x, y])

    print(f"Data has been exported to {filename}")


def Find_Depth(image_path, min_percentage=0.1, scale_length=5, export_name=None):
    """
    Load the image, fill holes before and after binarization, apply color-based binarization, 
    remove small clusters, crop the image, and connect the bottom of the clusters with a line.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image from", image_path)
        return

    # Step 1: Fill holes before binarization
    filled_before_binarization = fill_holes_before_binarization(image)

    # Step 2: Define the color range for binarization in HSV
    lower_hsv = (0, 0, 160)    # Example: Keep bright white regions
    upper_hsv = (180, 50, 255)  # Example: Allow only near-white shades

    # Step 3: Apply the color-based binarization
    binarized_image = binarize_by_color(
        filled_before_binarization, lower_hsv, upper_hsv)

    # Step 4: Fill holes after binarization
    filled_after_binarization = fill_holes_after_binarization(binarized_image)

    # Insert calculation of scale here
    # Step 5: Calculate the scaling factor using the binarized image
    scaling_factor = calculate_scaling_factor(
        image, scale_length)

    # Step 5: Remove small clusters and keep only large enough ones
    filtered_image = remove_small_clusters(
        filled_after_binarization, min_percentage)

    # Step 6: Find contours and select the clusters
    contours, _ = cv2.findContours(
        filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Crop the image based on the extreme edges of all clusters
    cropped_image = crop_to_extreme_clusters(filtered_image, contours)

    # Step 6: Find contours and select the clusters
    contours, _ = cv2.findContours(
        cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Connect the bottom of the clusters with a white line and set everything below it white
    bottom_connected = connect_bottom_of_clusters(cropped_image, contours)

    # Step 8: Remove all but the largest black cluster
    final_image = remove_all_but_largest_black_cluster(bottom_connected)

    xvals, yvals = test_find_topmost_distance(final_image, scaling_factor)

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), dpi=500)
    axs[1].plot(xvals, yvals, color='b')
    axs[1].set_title("Extracted Keyhole Contour")
    axs[1].set_xlabel("Length")
    axs[1].set_ylabel("Keyhole depth")
    axs[1].set_xlim(0, xvals[-1])

    # Add units to the axes 
    add_units_to_ticks(axs[1], axis='x', unit='mm')  # X-axis with 'mm'
    add_units_to_ticks(axs[1], axis='y', unit='mm')  # Y-axis with 'mm'

    # Plot the final image in the second subplot
    axs[0].imshow(final_image, cmap="gray")
    axs[0].axis('off')  # Turn off axis
    axs[0].set_title('Processed Image')

    rect = patches.Rectangle((0, 0), final_image.shape[1], final_image.shape[0],
                             linewidth=3, edgecolor="black", facecolor='none')  # Red frame
    axs[0].add_patch(rect)

    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.tight_layout()
    plt.show()
    
    # Exports data to 'depth_data.csv'
    if export_name != None:
      export_to_csv(xvals, yvals, filename=export_name)


if __name__ == '__main__':
    image_path = "1-1.png"
    Find_Depth(image_path, min_percentage=5, scale_length = 5, export_name = "depth_data.csv" )
