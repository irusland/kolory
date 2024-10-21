from PIL import Image

from PIL import Image
import numpy as np


image_path = 'image.webp'
image_path = 'forest2.webp'
image_path = 'images.jpeg'

import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image (assuming it's already read as a NumPy array)
img = cv2.imread(image_path)

# Reshape the image to a 2D array of pixels (each row represents a pixel, each column is the RGB channels)
pixels = img.reshape((-1, 3))

print(pixels.shape)
import numpy as np
import plotly.graph_objects as go

# Subset of the pixels for faster plotting (for example, 10,000 points)
sample_pixels = pixels[np.random.choice(pixels.shape[0], 50000, replace=False)]
# sample_pixels = pixels

# Create a 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=sample_pixels[:, 0],  # Red channel
    y=sample_pixels[:, 1],  # Green channel
    z=sample_pixels[:, 2],  # Blue channel
    mode='markers',
    marker=dict(
        size=2,
        color=sample_pixels / 255.0,  # Color according to RGB values
        opacity=0.8
    )
)])

# Set the labels for the axes
fig.update_layout(scene=dict(
    xaxis_title='Red',
    yaxis_title='Green',
    zaxis_title='Blue',
))

# Show the interactive plot
fig.show()


# exit()

# Apply K-Means clustering with k clusters (e.g., 5 colors)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

# Get the cluster centroids (the representative colors)
cluster_centers = kmeans.cluster_centers_.astype(int)

# Get the cluster labels for each pixel
labels = kmeans.labels_

# Replace each pixel value with its cluster center (color)
segmented_img = cluster_centers[labels]

# Reshape the clustered pixels back into the original image shape
segmented_img = segmented_img.reshape(img.shape)
# Convert the segmented image back to unsigned 8-bit integer type
segmented_img = segmented_img.astype(np.uint8)

# Now apply the color conversion
plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
plt.title(f'Segmented Image with {k} Colors')
plt.axis('off')
plt.show()
# Display the original and the segmented image
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
plt.title(f'Segmented Image with {k} Colors')
plt.axis('off')

plt.show()