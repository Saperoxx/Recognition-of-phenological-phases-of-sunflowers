import concurrent.futures
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Any

import numpy as np
from labelbox import Client
from labelbox.data.annotation_types import Label, Mask, Radio
from skimage import io
from tqdm import tqdm
import cv2

def download():
    client = Client(api_key='LABEL_API_KEY')

    project = client.get_project('PROJECT_ID')
    labels = project.label_generator()

    rgb_dir = Path('./data/sunflowers/images/')
    labels_dir = Path('./data/sunflowers/masks/')

    rgb_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    futures: List[Future] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for label in labels:
            label: Label

            image_name = Path(label.data.external_id)
            image_output_path = rgb_dir / image_name
            labels_output_path = labels_dir / f'{image_name.stem}.png'

            print(image_output_path)
            if not (image_output_path.exists() and labels_output_path.exists()):
                futures.append(executor.submit(download_image, label.data.url, image_output_path))
                futures.append(executor.submit(download_annotations, label.annotations, labels_output_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()


def download_image(url: str, output_path: Path):
    image = io.imread(url)
    io.imsave(str(output_path), image)

def is_ripe(mask):
    return str(mask['value']) == 'ripe_sunflower'

def is_ripe_85(mask):
    return str(mask['value']) == 'ripe_sunflower_85'

def is_unripe(mask):
    return str(mask['value']) == 'unripe_sunflower'

def download_annotations(annotations: List[Any], output_path: Path):
    #colors
    white = np.uint8((255, 255, 255))
    red = np.uint8((255, 0, 0))
    blue = np.uint8((0, 0, 255))
    green = np.uint8((0, 255, 0))
    orange = np.uint8((255, 100, 0))

    mask_data = None

    for annotation in annotations:
        if isinstance(annotation.extra, Radio):
            continue

        mask: Mask = annotation.extra
        annotation_data = io.imread(mask['instanceURI'])[..., :3]
        mask_data = np.zeros_like(annotation_data) if mask_data is None else mask_data

        if is_ripe(mask):
            mask_data = np.where(annotation_data == white, green, mask_data)

        if is_ripe_85(mask):
            mask_data = np.where(annotation_data == white, orange, mask_data)

        if is_unripe(mask):
            mask_data = np.where(annotation_data == white, red, mask_data)

    if mask_data is not None:
        io.imsave(str(output_path), mask_data, check_contrast=False)




if __name__ == '__main__':
    download()