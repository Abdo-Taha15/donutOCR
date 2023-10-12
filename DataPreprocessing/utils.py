import pandas as pd
import numpy as np
import requests
import random
import json
import shutil
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageOps

from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=True, lang="ch")


def download_file(url: str, name: str):
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    path = Path("dataset") / name
    with open(path, "wb") as f:
        response = requests.get(url)
        f.write(response.content)

    return path


def convert_pdf_to_image(filename: str | Path):
    # * returns back a list of images according to the pdf pages
    pdf_pages = convert_from_path(filename, 500)
    return pdf_pages


def resize_image(img: Image.Image):
    # * Resize the image while keeping its aspect ratio
    img.thumbnail((1000, 1000))

    # * Fix image orientation using exif info
    img = ImageOps.exif_transpose(img)


def create_dir(img: Image.Image, row: pd.Series, path: Path):
    results = ocr.ocr(np.asarray(img), det=False, rec=False, cls=True)
    orientation, conf = results[0][0]
    orientation = int(orientation)

    img_dir = {
        "file_name": None,
        "ground_truth": None,
        "rotated": None,
        "manual_edit": None,
    }
    if conf < 0.8:
        img_dir["manual_edit"] = True
        img_dir["rotated"] = False
    else:
        if orientation != 0:
            img.rotate(orientation, expand=True)
            img_dir["rotated"] = True
        else:
            img_dir["rotated"] = False
        img_dir["manual_edit"] = False
    img.save(path)
    img_dir["file_name"] = path.name
    img_dir["ground_truth"] = json.dumps(
        {"gt_parse": {"text_sequence": row["ocr_text"]}}
    )

    return img_dir


def split_data(seed: int | None = 42, dataset_path: str = ""):
    random.seed(seed)

    dataset_path = Path(dataset_path)

    if not os.path.exists(dataset_path / "train"):
        os.mkdir(str(dataset_path / "train"))
    if not os.path.exists(dataset_path / "test"):
        os.mkdir(str(dataset_path / "test"))
    if not os.path.exists(dataset_path / "validation"):
        os.mkdir(str(dataset_path / "validation"))

    data = []
    with open(dataset_path / "metadata.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)

    length = len(data)
    train_length = int(length * 0.7)
    val_length = int(length * 0.9)

    train_data = data[:train_length]
    val_data = data[train_length:val_length]
    test_data = data[val_length:]

    for file in train_data:
        shutil.move(
            src=dataset_path / file["file_name"],
            dst=dataset_path / "train" / file["file_name"],
        )

    with open(dataset_path / "train/metadata.jsonl", "w") as outfile:
        for entry in train_data:
            json.dump(entry, outfile)
            outfile.write("\n")

    for file in val_data:
        shutil.move(
            src=dataset_path / file["file_name"],
            dst=dataset_path / "validation" / file["file_name"],
        )

    with open(dataset_path / "validation/metadata.jsonl", "w") as outfile:
        for entry in val_data:
            json.dump(entry, outfile)
            outfile.write("\n")

    for file in test_data:
        shutil.move(
            src=dataset_path / file["file_name"],
            dst=dataset_path / "test" / file["file_name"],
        )

    with open(dataset_path / "test/metadata.jsonl", "w") as outfile:
        for entry in test_data:
            json.dump(entry, outfile)
            outfile.write("\n")

    os.remove(dataset_path / "metadata.jsonl")
