import torch
from PIL import Image

from rembg import remove


def predict(pretrained_model, img: Image, remove_background: bool = False):
    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    output = pretrained_model.inference(
        image=remove(img) if remove_background else img,
        prompt="<s_cord-v2>",
    )

    return output["predictions"][0]
