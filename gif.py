from PIL import Image
import os

# Specify the directory where your screenshots are stored
image_folder = "streamloop"

# Get all the image file names in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

# Sort the images (if needed, based on the file name)
images.sort()

def resize_with_aspect_ratio(image, base_width=600):
    """Resize the image while maintaining the aspect ratio."""
    w_percent = (base_width / float(image.size[0]))  # Calculate width percentage
    new_height = int((float(image.size[1]) * float(w_percent)))  # Calculate new height based on aspect ratio
    return image.resize((base_width, new_height), Image.Resampling.LANCZOS)

# Resize the images while maintaining the aspect ratio
frames = [resize_with_aspect_ratio(Image.open(os.path.join(image_folder, img))) for img in images]

# Reverse the frames to make the GIF loop back
frames_reversed = frames[::-1]

# Combine original and reversed frames
all_frames = frames + frames_reversed

# Save the frames as a compressed GIF with reduced colors (e.g., 128 colors)
gif_path = "output_compressed.gif"
all_frames[0].save(
    gif_path, format="GIF", append_images=all_frames[1:], save_all=True, duration=100, loop=0,
    optimize=True
)

print(f"Compressed GIF saved as {gif_path}")