import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans


st.title("Image Clustering with K-Means")
st.write(
    "Upload an image, choose the number of clusters (k), and explore the segmentation result and RGB scatter plot."
)

# Upload the image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Open the image file and convert it to RGB
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Ensure it's in RGB mode

    # Convert the image to a NumPy array
    img = np.array(img)

    col1, col2 = st.columns([1, 1])

    # Display the original image
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))

    # Select the number of clusters (k) using a slider
    # with col1:
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=21, value=3)

    # Apply K-Means clustering
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

    # Display the segmented image
    with col2:
        st.image(
            segmented_img, caption=f'Segmented Image with {k} Colors', use_column_width=True
        )

    # Subset the pixels for faster plotting (e.g., 10,000 points)
    sample_pixels = pixels[np.random.choice(pixels.shape[0], 10000, replace=False)]

    # Create a 3D scatter plot using Plotly
    fig = go.Figure(
        data=[go.Scatter3d(
            x=sample_pixels[:, 0],  # Red channel
            y=sample_pixels[:, 1],  # Green channel
            z=sample_pixels[:, 2],  # Blue channel
            mode='markers',
            marker=dict(
                size=2,
                color=sample_pixels / 255.0,  # Color according to RGB values
                opacity=0.8
            )
        )]
    )

    # Set the labels for the axes
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
        )
    )

    # Show the interactive 3D plot
    st.plotly_chart(fig)

# Footer
st.write("Made with ❤️ using Streamlit and K-Means clustering")