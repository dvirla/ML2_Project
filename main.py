import tensorflow as tf
import csv
from collections import defaultdict
import random
import numpy as np

random.seed = 42  # Fixing randomness


def parse_images_captions_file(path='Flickr8k_text/Flickr8k.token.txt'):
    """
    Reading the image-captions file
    :return: a dictionary {image_name: list of captions}
    """
    img_to_captions_dict = defaultdict(list)
    with open(path) as f:
        csvreader = csv.reader(f, delimiter='\t')
        for row in csvreader:
            img_caption, caption = row
            img = img_caption.split('#')[0]
            img_to_captions_dict[img].append(caption)
    return img_to_captions_dict


def main():
    img_to_captions_dict = parse_images_captions_file()
    # Splitting the image list into train, test sets
    images_list = list(img_to_captions_dict.keys())
    random.shuffle(images_list)
    train_images, test_images = images_list[:6000], images_list[6000:]