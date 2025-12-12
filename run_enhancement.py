"""Run Real-ESRGAN upscaling followed by GFPGAN face enhancement."""
import argparse
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale images and enhance faces.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the enhanced image")
    parser.add_argument(
        "--sr-model",
        default="RealESRGAN_x4plus.pth",
        help="Path to the Real-ESRGAN x4 model weights",
    )
    parser.add_argument(
        "--gfpgan-model",
        default="GFPGANv1.4.pth",
        help="Path to the GFPGAN model weights",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )
    return parser.parse_args()


def get_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def build_sr_upsampler(model_path: str, device: str) -> RealESRGANer:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=device == "cuda",
        device=device,
    )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    upsampler = build_sr_upsampler(args.sr_model, device)
    restorer = GFPGANer(
        model_path=args.gfpgan_model,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device,
    )

    input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if input_img is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input}")

    upscaled_img, _ = upsampler.enhance(input_img, outscale=4)
    _, _, restored_img = restorer.enhance(
        upscaled_img, has_aligned=False, only_center_face=False, paste_back=True
    )

    cv2.imwrite(args.output, restored_img)
    print(f"Saved enhanced image to {args.output}")


if __name__ == "__main__":
    main()
