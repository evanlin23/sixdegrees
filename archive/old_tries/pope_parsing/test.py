import os
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip
import numpy as np

# --- Configuration ---
IMAGE_FOLDER = "pope_images_indexed"
OUTPUT_VIDEO_FILE = "popes_timeline.mp4"
IMAGE_DURATION_SECONDS = 0.5  # Duration each image is shown
VIDEO_WIDTH = 720  # Width of the output video in pixels (e.g., 720 for 720p width)
FPS = 24 # Frames per second for the video

# Text Styling
try:
    # Try a common system font, adjust if not found or provide full path
    FONT_PATH = "roboto.ttf" # For Windows
    # On Linux, you might use: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # On macOS, you might use: "/System/Library/Fonts/Helvetica.ttc"
    FONT_SIZE = 40
except IOError:
    print(f"Warning: Font {FONT_PATH} not found. Using default PIL font. Text might look basic.")
    FONT_PATH = None # Pillow will use a default bitmap font
    FONT_SIZE = 40 # Default font might not scale well, adjust if needed

FONT_SIZE = 100
TEXT_COLOR = (	240, 255, 0)  # yellow
TEXT_OUTLINE_COLOR = (0, 0, 0) # Black
TEXT_OUTLINE_WIDTH = 5
TEXT_Y_OFFSET_FROM_BOTTOM = 300  # Pixels from the bottom edge
BACKGROUND_COLOR = (0, 0, 0) # white, for letterboxing if necessary

# --- Helper Function ---
def get_pope_name(filename):
    """Extracts pope name from filename like '1_Pius_IV.jpg' -> 'Pius IV'"""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    if len(parts) > 1:
        return " ".join(parts[1:])
    return base_name # Fallback

def add_text_to_image(pil_image, text, font_path, font_size, text_color, outline_color, outline_width, y_offset_from_bottom):
    """Adds centered text with an outline to the bottom of a PIL image."""
    draw = ImageDraw.Draw(pil_image)
    
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default() # Use if custom font failed
    except IOError:
        print(f"Error loading font: {font_path}. Using PIL default.")
        font = ImageFont.load_default()

    # Get text bounding box using textbbox for newer Pillow, textsize for older
    if hasattr(draw, 'textbbox'):
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else: # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)

    image_width, image_height = pil_image.size
    
    x = (image_width - text_width) / 2
    y = image_height - text_height - y_offset_from_bottom

    # Draw outline (stroke)
    if outline_width > 0 and outline_color:
        for i in range(-outline_width, outline_width + 1):
            for j in range(-outline_width, outline_width + 1):
                if i != 0 or j != 0: # don't draw center pixel yet
                    draw.text((x + i, y + j), text, font=font, fill=outline_color)
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)
    return pil_image

# --- Main Video Creation Logic ---
def create_video():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
        return

    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=lambda x: int(x.split('_')[0])) # Sort by the leading number

    if not image_files:
        print(f"No images found in '{IMAGE_FOLDER}'.")
        return

    print(f"Found {len(image_files)} images. Processing...")

    processed_frames = []
    
    # First pass: determine the maximum height after scaling to VIDEO_WIDTH
    # This is to ensure all images fit into a consistent video frame height
    max_scaled_height = 0
    print("Calculating video dimensions...")
    for filename in image_files:
        try:
            img_path = os.path.join(IMAGE_FOLDER, filename)
            with Image.open(img_path) as img:
                original_width, original_height = img.size
                if original_width == 0: continue # Skip invalid images
                aspect_ratio = original_height / original_width
                scaled_height = int(VIDEO_WIDTH * aspect_ratio)
                if scaled_height > max_scaled_height:
                    max_scaled_height = scaled_height
        except Exception as e:
            print(f"Warning: Could not process {filename} for dimension check: {e}")
            continue
    
    if max_scaled_height == 0:
        print("Error: Could not determine video dimensions. No valid images processed.")
        return
        
    VIDEO_HEIGHT = max_scaled_height
    print(f"Video dimensions set to: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

    # Second pass: process images, scale, add text, and create frames
    for i, filename in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {filename}")
        try:
            pope_name = get_pope_name(filename)
            img_path = os.path.join(IMAGE_FOLDER, filename)
            
            with Image.open(img_path).convert("RGB") as img: # Convert to RGB
                original_width, original_height = img.size
                
                # Scale image to VIDEO_WIDTH, maintaining aspect ratio
                aspect_ratio = original_height / original_width
                new_height = int(VIDEO_WIDTH * aspect_ratio)
                resized_img = img.resize((VIDEO_WIDTH, new_height), Image.Resampling.LANCZOS)
                
                # Create a new image canvas with the final video dimensions
                # and paste the resized image onto it (vertically centered)
                frame_canvas = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), BACKGROUND_COLOR)
                
                y_paste_offset = (VIDEO_HEIGHT - new_height) // 2
                frame_canvas.paste(resized_img, (0, y_paste_offset))
                
                # Add pope name text
                frame_with_text = add_text_to_image(
                    frame_canvas,
                    pope_name,
                    FONT_PATH,
                    FONT_SIZE,
                    TEXT_COLOR,
                    TEXT_OUTLINE_COLOR,
                    TEXT_OUTLINE_WIDTH,
                    TEXT_Y_OFFSET_FROM_BOTTOM
                )
                
                # Convert PIL image to NumPy array for MoviePy
                processed_frames.append(np.array(frame_with_text))
                
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            continue # Skip this image

    if not processed_frames:
        print("No frames were processed successfully. Video not created.")
        return

    # Create video clip from image sequence
    print("Creating video clip...")
    video_clip = ImageSequenceClip(processed_frames, fps=FPS, durations=[IMAGE_DURATION_SECONDS]*len(processed_frames))
    
    # Write the video to a file
    print(f"Writing video to {OUTPUT_VIDEO_FILE}...")
    try:
        video_clip.write_videofile(OUTPUT_VIDEO_FILE, codec='libx264', audio=False, threads=4, logger='bar')
        print("Video created successfully!")
    except Exception as e:
        print(f"Error writing video file: {e}")
        print("You might need to install ffmpeg and ensure it's in your system PATH.")
        print("Or, try a different codec like 'mpeg4'.")

if __name__ == "__main__":
    create_video()