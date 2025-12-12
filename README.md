# Face Enhancement

A simple pipeline for local Real-ESRGAN upscaling followed by GFPGAN face restoration.

## Installation

1. Create and activate a Python environment (Python 3.8+ is recommended).
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If you plan to use a GPU, ensure your CUDA toolkit matches the installed PyTorch build.

## Download model weights

- Real-ESRGAN x4 model: download `RealESRGAN_x4plus.pth` from the [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).
- GFPGAN model: download `GFPGANv1.4.pth` from the [GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases).

Place both files in the repository root or supply their paths when running the script.

## Usage

Run the enhancement script after downloading the model weights:

```bash
python run_enhancement.py input.jpg output.png \
  --sr-model RealESRGAN_x4plus.pth \
  --gfpgan-model GFPGANv1.4.pth \
  --device auto
```

- The script first upsamples the image 4× with Real-ESRGAN and then applies GFPGAN face enhancement.
- Set `--device cuda` to force GPU inference or `--device cpu` for CPU-only inference.
- Input and output paths can be absolute or relative; output will be overwritten if it exists.

## Notes

- Large images may require tiling parameters; adjust them in `run_enhancement.py` if you encounter GPU memory errors.
- For batch processing, wrap the script call in your own loop or extend the script to accept directories.
