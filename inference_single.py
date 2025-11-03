import os
import argparse
import torch
from PIL import Image
import numpy as np
import cv2

# Import CatVTON pipeline and cloth masker
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker, vis_mask


def generate_person_mask(person_img_path, output_mask_path):
    """
    Generate a real person mask using CatVTON's AutoMasker (DensePose + parsing).
    This replaces the dummy white mask and helps preserve face/background.
    """
    print("🧩 Generating real person mask using DensePose + ClothMasker...")
    masker = AutoMasker(device="cuda" if torch.cuda.is_available() else "cpu")

    # Run inference to get mask
    mask_tensor = masker(person_img_path)
    mask = mask_tensor.squeeze().cpu().numpy() * 255
    mask = mask.astype(np.uint8)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    cv2.imwrite(output_mask_path, mask)

    print(f"✅ Saved real mask to {output_mask_path}")
    return output_mask_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_path", type=str, required=True, help="Path to person image")
    parser.add_argument("--cloth_path", type=str, required=True, help="Path to cloth image")
    parser.add_argument("--output_dir", type=str, default="./results_single", help="Output directory")
    parser.add_argument("--resolution", type=str, default="768x1024", help="Output resolution (WxH)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    args = parser.parse_args()

    width, height = map(int, args.resolution.lower().split("x"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Generate realistic mask for the person
    mask_path = os.path.join(args.output_dir, "person_mask.png")
    generate_person_mask(args.person_path, mask_path)

    # Step 2: Load pipeline
    print("🔹 Loading CatVTON pipeline weights...")

    base_ckpt = "runwayml/stable-diffusion-v1-5"
    attn_ckpt = "/root/.cache/huggingface/hub/models--zhengchong--CatVTON/snapshots/2969fcf85fe62f2036605716f0b56f0b81d01d79"
    attn_ckpt_version = "vitonhd"

    pipeline = CatVTONPipeline(
        base_ckpt=base_ckpt,
        attn_ckpt=attn_ckpt,
        attn_ckpt_version=attn_ckpt_version,
        device=args.device,
        skip_safety_check=True,
        use_tf32=True,
    )

    # Step 3: Load inputs
    person_img = Image.open(args.person_path).convert("RGB").resize((width, height))
    cloth_img = Image.open(args.cloth_path).convert("RGB").resize((width, height))
    mask_img = Image.open(mask_path).convert("L").resize((width, height))

    # Step 4: Run inference
    torch.manual_seed(args.seed)
    print("🚀 Running CatVTON inference with real mask...")
    result = pipeline(
        image=person_img,
        condition_image=cloth_img,
        mask=mask_img,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=5.0,
    )

    # Step 5: Save output
    out_path = os.path.join(args.output_dir, "tryon_result.jpg")
    result[0].save(out_path)
    print(f"✅ Saved try-on result to: {out_path}")

    # Step 6: Show result inline (for Colab)
    try:
        from IPython.display import Image as ColabImage, display
        display(ColabImage(filename=out_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
