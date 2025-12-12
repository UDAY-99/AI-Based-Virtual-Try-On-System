from pathlib import Path
import sys
import os
from PIL import Image

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

import argparse


def scale_mask(mask: Image.Image, scale: float) -> Image.Image:
    """
    Scale the clothing mask around its bounding box center.

    scale < 1.0 -> tighter (smaller mask)
    scale > 1.0 -> looser (bigger mask)
    scale = 1.0 -> unchanged
    """
    if abs(scale - 1.0) < 1e-3:
        return mask

    if mask.mode != "L":
        # ensure single-channel mask
        mask = mask.convert("L")

    w, h = mask.size
    bbox = mask.getbbox()
    if bbox is None:
        # no foreground, nothing to scale
        return mask

    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0

    if bw <= 0 or bh <= 0:
        return mask

    # center of the clothing area
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0

    # new width/height
    new_bw = max(1, int(bw * scale))
    new_bh = max(1, int(bh * scale))

    # crop original mask region
    crop = mask.crop((x0, y0, x1, y1))
    crop_resized = crop.resize((new_bw, new_bh), resample=Image.NEAREST)

    # place resized crop back onto a blank canvas, centered
    new_mask = Image.new("L", (w, h), 0)

    # target coordinates
    new_x0 = int(round(cx - new_bw / 2.0))
    new_y0 = int(round(cy - new_bh / 2.0))
    new_x1 = new_x0 + new_bw
    new_y1 = new_y0 + new_bh

    # handle borders: adjust src region if paste box goes out of image
    src_x0, src_y0 = 0, 0
    src_x1, src_y1 = new_bw, new_bh

    if new_x0 < 0:
        src_x0 = -new_x0
        new_x0 = 0
    if new_y0 < 0:
        src_y0 = -new_y0
        new_y0 = 0
    if new_x1 > w:
        src_x1 -= (new_x1 - w)
        new_x1 = w
    if new_y1 > h:
        src_y1 -= (new_y1 - h)
        new_y1 = h

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        # degenerate after clipping
        return mask

    crop_cropped = crop_resized.crop((src_x0, src_y0, src_x1, src_y1))
    new_mask.paste(crop_cropped, (new_x0, new_y0))

    return new_mask


# ----------------- argument parsing (copied from run_ootd.py + new arg) -----------------

parser = argparse.ArgumentParser(description="run ootd with simple fit control")

parser.add_argument("--gpu_id", "-g", type=int, default=0, required=False)
parser.add_argument("--model_path", type=str, default="", required=True)
parser.add_argument("--cloth_path", type=str, default="", required=True)
parser.add_argument("--model_type", type=str, default="hd", required=False)
parser.add_argument("--category", "-c", type=int, default=0, required=False)

parser.add_argument("--scale", type=float, default=2.0, required=False)
parser.add_argument("--step", type=int, default=20, required=False)
parser.add_argument("--sample", type=int, default=4, required=False)
parser.add_argument("--seed", type=int, default=-1, required=False)

parser.add_argument("--out_dir", type=str, default="./images_output", required=False)


# NEW: personalized fitting parameter
parser.add_argument(
    "--fit_scale",
    type=float,
    default=1.0,
    required=False,
    help="Cloth fit scale: <1.0 tighter, >1.0 looser, 1.0 = normal.",
)

args = parser.parse_args()
out_dir = args.out_dir


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)

category_dict = ["upperbody", "lowerbody", "dress"]
category_dict_utils = ["upper_body", "lower_body", "dresses"]

model_type = args.model_type  # "hd" or "dc"
category = args.category      # 0:upperbody; 1:lowerbody; 2:dress

cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed
fit_scale = args.fit_scale

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be 'hd' or 'dc'!")


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)


    if model_type == "hd" and category != 0:
        raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")

    # 1) load & resize images
    cloth_img = Image.open(cloth_path).convert("RGB").resize((768, 1024))
    model_img = Image.open(model_path).convert("RGB").resize((768, 1024))

    # 2) pose & parsing on smaller version (same as original)
    model_img_small = model_img.resize((384, 512))
    keypoints = openpose_model(model_img_small)
    model_parse, _ = parsing_model(model_img_small)

    # 3) get mask & gray base
    mask, mask_gray = get_mask_location(
        model_type,
        category_dict_utils[category],
        model_parse,
        keypoints,
    )

    # 4) resize masks to 768x1024
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

    # 5) APPLY PERSONALIZED FIT: scale the mask
    #    fit_scale < 1.0 -> tighter; fit_scale > 1.0 -> looser
    if fit_scale is not None and abs(fit_scale - 1.0) > 1e-3:
        mask = scale_mask(mask, fit_scale)

    # 6) composite masked image
    masked_vton_img = Image.composite(mask_gray, model_img, mask)

    # save mask for inspection
    masked_vton_img.save(os.path.join(out_dir, "mask_fit.jpg"))


    # 7) run OOTDiffusion model
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    # 8) save outputs (separate prefix so you know it's the fit version)
    image_idx = 0
    for image in images:
        image.save(
        os.path.join(
        out_dir,
        "out_fit_" + model_type + "_" + str(image_idx) + ".png"
        )
        )
        image_idx += 1
