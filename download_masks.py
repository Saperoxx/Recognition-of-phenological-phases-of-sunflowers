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
    client = Client(api_key='API_KEY')

    project = client.get_project('PROJECT_KEY')
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
    white = (255, 255, 255)
    mask_data = None

    for annotation in annotations:
        if isinstance(annotation.extra, Radio):
            continue

        mask: Mask = annotation.extra
        annotation_data = io.imread(mask['instanceURI'])[..., :3]
        mask_data = np.zeros_like(annotation_data) if mask_data is None else mask_data
        mask_data = np.where(annotation_data == white, np.uint8(white), mask_data)

    if mask_data is not None:
        mask_data = cv2.cvtColor(mask_data, cv2.COLOR_RGB2GRAY)
        io.imsave(str(output_path), mask_data, check_contrast=False)
        print(str(output_path))

if __name__ == '__main__':
    download()