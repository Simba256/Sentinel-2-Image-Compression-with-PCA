# Import necessary libraries
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to read a single band from a raster file
def read_band(band_path):
    # Open the raster file and read a specific band
    with rasterio.open(band_path) as src:
        return src.read(1)

# Function to get paths of all band files in a folder
def get_band_paths(folder_path):
    # Get all files in the folder and filter for .jp2 files (assuming they're raster bands)
    all_files = os.listdir(folder_path)
    return [os.path.join(folder_path, file) for file in all_files if file.endswith('.jp2')]

# Function to load all bands given their file paths
def load_bands(band_paths):
    # Read each band and store them in a list
    return [read_band(path) for path in band_paths]

# Function to select a crop area from a stacked image
# Allows the user to interactively select the area using matplotlib
def select_crop_area(stacked):
    # Define variables to store click coordinates and cropping status
    ix, iy, end_row, end_col, cropping = [0], [0], [0], [0], [False]

    # Function to handle mouse clicks on the displayed image
    def on_click(event):
        # Left-click to start selecting and finish selecting the crop area
        if event.button == 1:
            if not cropping[0]:
                # Initial position when the user starts selecting
                ix[0], iy[0] = event.xdata, event.ydata
                print(f"Initial position: x={ix[0]}, y={iy[0]}")
                cropping[0] = True
            else:
                # Final position when the user finishes selecting
                end_row[0], end_col[0] = int(event.ydata), int(event.xdata)
                print(f"Final position: x={end_col[0]}, y={end_row[0]}")
                fig.canvas.mpl_disconnect(cid)  # Disconnect the click event
                plt.close(fig)  # Close the displayed image

    # Initialize variables and display the stacked image for selection
    cropping[0] = False
    fig, ax = plt.subplots()
    ax.imshow(np.sum(stacked, axis=2), cmap='gray')  # Display the stacked image
    ax.set_title('Click to select crop area (left-click to start, left-click to finish)')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Connect the mouse click event to the function
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    # Return the selected crop area coordinates
    return int(ix[0]), int(iy[0]), end_row[0], end_col[0]

# Function to crop an image based on selected coordinates
def crop_image(stacked, start_row, start_col, end_row, end_col):
    return stacked[start_row:end_row, start_col:end_col, :]

# Function to perform Principal Component Analysis (PCA) on the cropped image
def perform_pca(cropped_image, num_components):
    # Reshape the image data for PCA
    rows, cols, num_bands = cropped_image.shape
    reshaped = cropped_image.reshape(rows * cols, num_bands)
    
    # Standardize the data (mean centering and scaling)
    mean = np.mean(reshaped, axis=0)
    std_dev = np.std(reshaped, axis=0)
    standardized = (reshaped - mean) / std_dev

    
    # Calculate covariance matrix, eigenvalues, and eigenvectors
    cov_matrix_standardized = np.cov(standardized, rowvar=False)
    eigenvalues_standardized, eigenvectors_standardized = np.linalg.eigh(cov_matrix_standardized)
    
    # Sort eigenvectors based on eigenvalues
    sorted_indices_standardized = np.argsort(eigenvalues_standardized)[::-1]
    sorted_eigenvectors_standardized = eigenvectors_standardized[:, sorted_indices_standardized]
    
    # Select top components and project the data
    selected_components_standardized = sorted_eigenvectors_standardized[:, :num_components]
    projected_standardized = np.dot(standardized, selected_components_standardized)
    
    # Reshape the projected data into an image
    projected_image_standardized = projected_standardized.reshape(rows, cols, num_components)
    return projected_image_standardized, selected_components_standardized, standardized, mean

# Function to apply a threshold to the PCA results
def apply_threshold(projected_image_standardized, threshold_value):
    return np.where(np.abs(projected_image_standardized) > threshold_value, projected_image_standardized, 0)

# Function to reconstruct the image from PCA results
def reconstruct_image(projected_standardized, selected_components_standardized, rows, cols, num_bands):
    # Take transpose of components to perform inverse projection to image
    reconstructed_standardized = np.dot(projected_standardized, selected_components_standardized.T)
    reconstructed_image_standardized = reconstructed_standardized.reshape(rows, cols, num_bands)
    return reconstructed_image_standardized

# Function to calculate reconstruction error between original and reconstructed images
def calculate_reconstruction_error(standardized, reconstructed_standardized):
    # Calculate mean squared error between original and reconstructed data
    if len(standardized.shape) == 3:
        rows, cols, num_bands = standardized.shape
    elif len(standardized.shape) == 2:
        rows, cols = standardized.shape
        num_bands = 1  # Assume single band if only rows and columns are found
    else:
        raise ValueError("Unexpected shape for standardized data")

    flattened_standardized = standardized.reshape(rows * cols, num_bands)
    reconstructed_reshaped = reconstructed_standardized.reshape(rows * cols, num_bands)
    
    return np.sum((flattened_standardized - reconstructed_reshaped) ** 2) / (rows * cols * num_bands)

# Function to plot different images
def plot_images(original, after_pca, reconstructed):
    # Display original, PCA-transformed, and reconstructed images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    images = [original, after_pca, reconstructed]
    titles = ['Original Cropped Image', 'Image after PCA without Thresholding', 'Reconstructed Cropped Image']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(np.sum(img, axis=2), cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to calculate information loss with varying number of components
def calculate_information_loss(components, standardized_data, original_shape):
  """
  Calculates the information loss between the original and reconstructed images,
  and also returns the percentage loss.

  Args:
    components: Matrix containing the selected principal components.
    standardized_data: Standardized data of the original cropped image.
    original_shape: Original dimensions (rows, cols) of the cropped image.

  Returns:
    mse: Mean squared error between original and reconstructed images.
    percentage_loss: Information loss as a percentage.
  """

  # Perform projection and reconstruction
  projected_standardized = np.dot(standardized_data, components)
  reconstructed_standardized = np.dot(projected_standardized, components.T)
  reconstructed_image = reconstructed_standardized.reshape(original_shape)

  # Calculate MSE and original image variance
  standardized_reshaped = standardized_data.reshape(original_shape)
  mse = np.sum((standardized_reshaped - reconstructed_image) ** 2) / standardized_reshaped.size
  original_image_variance = np.var(standardized_data)

  # Calculate percentage loss
  percentage_loss = (mse / original_image_variance) * 100

  return mse, percentage_loss

# Function to analyze reconstruction error for different numbers of components
def perform_error_analysis(sorted_eigenvectors_standardized, standardized, cropped_image_shape, max_components_to_test):
    error_list = []
    percentage_loss_list = []
    
    # Test different numbers of components and calculate reconstruction error
    for n in range(1, max_components_to_test + 1):
        selected_components_standardized = sorted_eigenvectors_standardized[:, :n]
        error, percentage_loss = calculate_information_loss(selected_components_standardized, standardized, cropped_image_shape)
        error_list.append(error)
        percentage_loss_list.append(percentage_loss)
        print(f"Number of Components: {n}, Reconstruction Error: {error}, Percentage loss: {percentage_loss}%")
    
    return error_list, percentage_loss_list

# Function to plot information loss against number of components
def plot_information_loss(error_list, percentage_loss_list, max_components_to_test):
  """
  Plots the information loss as both mean squared error and percentage
  against the number of principal components, side-by-side.

  Args:
    error_list: List of mean squared error values for different components.
    percentage_loss_list: List of percentage information loss values
      for different components.
    max_components_to_test: Maximum number of components tested.
  """
  percentage_info_recovered = [100 - percentage_loss for percentage_loss in percentage_loss_list]
  # Create the figure and subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  # Plot mean squared error
  ax1.plot(range(1, max_components_to_test + 1), error_list, marker='o', linestyle='--', color='b', label="MSE")
  ax1.set_xlabel('Number of Principal Components')
  ax1.set_ylabel('Mean Squared Error')
  ax1.set_title('Reconstruction Error vs. Number of Components')
  ax1.grid(True)

  # Plot percentage loss
  ax2.plot(range(1, max_components_to_test + 1), percentage_info_recovered, marker='o', linestyle='--', color='g', label="Percentage Recovered")
  ax2.set_xlabel('Number of Principal Components')
  ax2.set_ylabel('Information Recovered (%)')
  ax2.set_title('Information Recovered (%) vs. Number of Components')
  ax2.grid(True)

  # Add legend
  ax2.legend()

  # Adjust layout and display the plot
  plt.tight_layout()
  plt.show()


# Function to recommend optimal number of components based on error analysis
def recommend_components(error_list):
    min_error_idx = np.argmin(error_list)
    optimal_components = min_error_idx + 1
    min_error = error_list[min_error_idx]
    
    print(f"\nRecommended Number of Principal Components: {optimal_components} with Reconstruction Error: {min_error}")

def visualize_bands(image):
    num_bands = image.shape[2]
    
    # Calculate the number of rows needed based on 4 images per row
    num_rows = (num_bands + 3) // 4  # Ceiling division to get the total rows
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, num_rows * 4))
    
    # Display the specified number of bands separately in a grid layout
    for i, ax in enumerate(axes.flat):
        if i < num_bands:
            band = image[:, :, i]
            ax.imshow(band, cmap='gray')
            ax.set_title(f'Band {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide empty subplots
    
    plt.tight_layout()
    plt.show()



# Main function to orchestrate the entire process
def main():
    folder_path = r"D:\Sentinel2bands"  # Path to folder containing band images0231119T074141.SAFE\GRANULE\L2A_T42RXU_A035009_20231119T060130\IMG_DATA\R60m"  # Path to folder containing band images
    band_paths = get_band_paths(folder_path)  # Get paths of all band images
    bands = load_bands(band_paths)  # Load all bands
    stacked = np.stack(bands, axis=-1)  # Stack bands into a single image
    
    # Select an area of interest in the image
    start_row, start_col, end_row, end_col = select_crop_area(stacked)
    cropped_image = crop_image(stacked, start_row, start_col, end_row, end_col)  # Crop the selected area
    
    print(f"Number of bands = {cropped_image.shape[2]}")

    num_components = int(input("Enter the number of components for PCA: ")) # Number of components for PCA
    projected_image_standardized, selected_components_standardized, standardized, mean = perform_pca(cropped_image, num_components)
    
    threshold_value = 0.5  # Threshold value for applying on PCA results
    projected_image_thresholded = apply_threshold(projected_image_standardized, threshold_value)
    
    # Reconstruct the image using PCA results
    reconstructed_image_standardized = reconstruct_image(projected_image_thresholded, selected_components_standardized, 
                                                    cropped_image.shape[0], cropped_image.shape[1], cropped_image.shape[2])
    
    # Calculate reconstruction error between original and reconstructed images
    reconstruction_error = calculate_reconstruction_error(standardized, reconstructed_image_standardized)
    print(f"Reconstruction Error: {reconstruction_error}")
    
    # Plot various images for comparison
    plot_images(cropped_image, projected_image_standardized, reconstructed_image_standardized)
    
    # Visualize individual bands of the cropped image
    visualize_bands(cropped_image)

    # Perform error analysis for different numbers of components
    error_list, percentage_loss_list = perform_error_analysis(selected_components_standardized, standardized, cropped_image.shape, num_components)
    plot_information_loss(error_list, percentage_loss_list, num_components)  # Plot information loss against number of components
    
    # Recommend the optimal number of components based on error analysis
    recommend_components(error_list)

if __name__ == "__main__":
    main()  

