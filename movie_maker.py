from pathlib import Path
from PIL import Image
import cv2
import re


def crop_images_in_dir(directory, crop_rect):
    """
    Crops all PNG images directly in the specified directory (not including subdirectories)
    to the given rectangle and overwrites them.

    Parameters:
        directory (str or Path): Directory containing PNG images.
        crop_rect (tuple): Crop rectangle (left, upper, right, lower).
    """
    dir_path = Path(directory)

    for img_path in dir_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() == '.png':
            with Image.open(img_path) as img:
                cropped_img = img.crop(crop_rect)
                cropped_img.save(img_path)
                print(f"Cropped and saved: {img_path.name}")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def make_movie_from_images(directory, output_movie_path, fps=25):
    """
    Creates a high-quality movie from PNG images in the specified directory, sorting images naturally.

    Parameters:
        directory (str or Path): Directory containing PNG images.
        output_movie_path (str or Path): Path to save the output movie file.
        fps (int): Frames per second for the movie. Default is 25.
    """
    dir_path = Path(directory)
    image_paths = sorted(
        [img for img in dir_path.iterdir() if img.suffix.lower() == '.png'],
        key=lambda x: natural_sort_key(x.stem)
    )

    if not image_paths:
        print("No images found in the directory.")
        return

    first_frame = cv2.imread(str(image_paths[0]))
    height, width, layers = first_frame.shape
    video = cv2.VideoWriter(str(output_movie_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        video.write(frame)

    video.release()
    print(f"Movie created at {output_movie_path}")
    
directory = "/home/shared/00_data/modes_3/Images/"
output_movie_path = "/home/shared/00_data/modes_3/mode_1.mp4"
crop_images_in_dir(directory, (1000, 60, 1950, 1000))
# make_movie_from_images(directory, output_movie_path, fps=2)