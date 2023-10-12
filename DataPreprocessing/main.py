from dotenv import load_dotenv
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import json

from utils import (
    download_file,
    convert_pdf_to_image,
    resize_image,
    create_dir,
    split_data,
)

load_dotenv()

SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "pdf", "JPG", "PNG", "PDF", "JPEG"]

# * Read CSV data
df = pd.read_csv("ocr_requests.csv")
text_table = pd.read_csv("ocr_texts.csv")

# * Append ocr_text to images
df["ocr_text"] = [
    eval(row["ocr_text"])
    for _, row in text_table.iterrows()
    if (row["id"] == r["id"].item() for _, r in df.iterrows())
]

# * Take n of random images from the dataset
df = df.sample(n=100, random_state=42)

# * Remove duplicates
df.drop_duplicates(subset=["file_hash"], inplace=True)

# * Add extension column to the dataframe to check it
df["extension"] = df["url"].apply(lambda row: row.split(".")[-1])

dataset = []
for idx, row in df.iterrows():
    if row["extension"] in SUPPORTED_EXTENSIONS:
        url = os.environ["BASE_URL"] + f"/{row['url']}"
        path: Path = download_file(url, name=f"{row['id']}.{row['extension']}")
        if row["extension"] in ["pdf", "PDF"]:
            pages = convert_pdf_to_image(path)
            if len(pages) > 1:
                path.unlink()
                continue
            else:
                for page in pages:
                    resize_image(page)
                    dataset.append(
                        create_dir(
                            page.convert("RGB"),
                            row,
                            Path(str(path).split(".")[0] + ".jpg"),
                        )
                    )
                path.unlink()
        else:
            img = Image.open(path)
            resize_image(img)
            dataset.append(
                create_dir(
                    img.convert("RGB"), row, Path(str(path).split(".")[0] + ".jpg")
                )
            )
            if row["extension"] != "jpg":
                path.unlink()


with open("dataset/metadata.jsonl", "w") as outfile:
    for entry in dataset:
        json.dump(entry, outfile)
        outfile.write("\n")

split_data(seed=42, dataset_path="dataset")
