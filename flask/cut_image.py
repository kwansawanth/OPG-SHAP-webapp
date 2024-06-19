# cut_image.py
from PIL import Image
import sys
import os

def cut_image(image_path, left_filename):
    img = Image.open(image_path)
    width, height = img.size
    frac = 0.6

    # Crop 60% from the left of the image
    crop_left_width = int(width * frac)
    cropped_left = img.crop((0, 0, crop_left_width, height))
    cropped_left.save(left_filename)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python cut_image.py <image_path> <left_image_output_path> ")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f'print image path from cut_image.py: {image_path}')
    left_image_output_path = sys.argv[2]
    print(f'print left image path from cut_image.py: {left_image_output_path}')

    cut_image(image_path, left_image_output_path)