import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans


def clear_pixels_choice():
    st.session_state.pixels_choice = None


uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "webp"], on_change=clear_pixels_choice
)


if uploaded_file is not None:
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=21, value=3)

    img = Image.open(uploaded_file)
    img = img.convert("RGB")

    img = np.array(img)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    pixels = img.reshape((-1, 3))


    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    cluster_centers = kmeans.cluster_centers_.astype(int)

    labels = kmeans.labels_

    segmented_img = cluster_centers[labels]

    segmented_img = segmented_img.reshape(img.shape)

    segmented_img = segmented_img.astype(np.uint8)

    with col2:
        st.image(
            segmented_img, caption=f'Segmented Image with {k} Colors',
            use_column_width=True
        )
    if st.session_state.pixels_choice is None:
        st.session_state.pixels_choice = np.random.choice(pixels.shape[0], 10000, replace=False)
    sample_pixels = pixels[st.session_state.pixels_choice]
    sample_labels = labels[st.session_state.pixels_choice]
    cluster_rgb_values = cluster_centers[sample_labels]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=sample_pixels[:, 0],
            y=sample_pixels[:, 1],
            z=sample_pixels[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=sample_pixels / 255.0,
                opacity=0.8
            )
        )]
    )

    camera = dict(
        eye=dict(x=1, y=-1, z=1),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            camera=camera
        ),
        title='Pixel colors',
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig)

    fig2 = go.Figure(
        data=[go.Scatter3d(
            x=sample_pixels[:, 0],
            y=sample_pixels[:, 1],
            z=sample_pixels[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=cluster_rgb_values / 255.0,
                opacity=0.8
            )
        )]
    )

    fig2.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            camera=camera
        ),
        title='Clustered colors',
    )

    with col2:

        st.plotly_chart(fig2)

st.title("Image Clustering with K-Means")
st.write(
    "Upload an image, choose the number of clusters (k), and explore the segmentation result and RGB scatter plot."
)
st.markdown("Made by [irusland](http://github.com/irusland) using Streamlit and K-Means clustering")
