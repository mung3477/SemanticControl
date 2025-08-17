import os
from typing import Tuple, TypedDict

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw


class TextDraw_Options(TypedDict):
    fill: Tuple

def image_grid(imgs, rows, cols, caption="", options: TextDraw_Options = { "fill" : (0, 0, 0)}):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    # https://www.geeksforgeeks.org/adding-text-on-image-using-python-pil/
    I1 = ImageDraw.Draw(grid)
    I1.text((10, 10), caption, fill=options['fill'])
    return grid

def calc_diff(img1: Image.Image , img2: Image.Image, name: str):
    assert img1.size == img2.size, f"{img1.size} does not match with {img2.size}"

    diff = ImageChops.subtract(img1, img2)

    grid = image_grid(
        [
                img1.resize((256, 256), resample=Image.NEAREST),
                img2.resize((256, 256), resample=Image.NEAREST),
                diff.resize((256, 256), resample=Image.NEAREST)
        ],
        rows=1,
        cols=3,
        caption=name,
        options={ "fill": (255, 255, 255) }
    )
    grid.save(f"{name}.png")
    print(f"attention difference were saved at {name}.png")

def gray2colormap(gray):
    gray_numpy = (gray * 255).byte().numpy()
    colormap = cv2.applyColorMap(gray_numpy, cv2.COLORMAP_JET)
    return colormap
