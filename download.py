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
    client = Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGV4NWd3YTkwbzdhMDc0M2RjZXdmM3Z4Iiwib3JnYW5pemF0aW9uSWQiOiJjbGV4NWd3ODMwbzc5MDc0M2ZwMzg4dno2IiwiYXBpS2V5SWQiOiJjbGYzdHdiNW4xY3drMDgyczByb2VmaHgyIiwic2VjcmV0IjoiODI3MTIwZjc1YTVhNTFlMWY0NWQyY2YzNmE0YzEzYTAiLCJpYXQiOjE2Nzg1MzA3OTUsImV4cCI6MjMwOTY4Mjc5NX0.mQKPRfOSpgQ-uhd3vsKvQs7cZXE0A5mw5mtTBI3U54Y')

    project = client.get_project('clezrgpm711mf07w63ev175kn')
    # project = client.get_project('clf3vdjfi0kjc073thlk7as7c')
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


def download_annotations(annotations: List[Any], output_path: Path):
    mask_data = None
    for annotation in annotations:
        if isinstance(annotation.extra, Radio):
            continue

        mask: Mask = annotation.extra
        # mask: Mask = annotation.value
        # annotation_data = io.imread(mask.mask.url)[..., :3]
        annotation_data = io.imread(mask['instanceURI'])[..., :3]
        print(mask)
        # if str(mask['value']) == 'ripe_sunflower':
        mask_data = np.zeros_like(annotation_data) if mask_data is None else mask_data
        mask_data = np.where(annotation_data == (255, 255, 255), np.uint8((0, 255, 0)), mask_data)
        # elif str(mask['value']) == 'ripe_sunflower_85%':
        #     mask_data = np.zeros_like(annotation_data) if mask_data is None else mask_data
        #     mask_data = np.where(annotation_data == (240, 180, 0), np.uint8((0, 255, 0)), mask_data)
        # elif str(mask['value']) == 'unripe_sunflower':
        #     print('good')
        #     mask_data = np.zeros_like(annotation_data) if mask_data is None else mask_data
        #     mask_data = np.where(annotation_data == (255, 0, 0), np.uint8((0, 255, 0)), mask_data)

    if mask_data is not None:
        io.imsave(str(output_path), mask_data, check_contrast=False)


if __name__ == '__main__':
    download()