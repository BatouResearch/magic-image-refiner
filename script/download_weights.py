from diffusers import ControlNetModel, DiffusionPipeline, AutoencoderKL, AutoPipelineForInpainting
import os
import subprocess
import tarfile
import torch

SD15_WEIGHTS = "weights"
INPAINT_WEIGHTS = "inpaint-cache"
TILE_CACHE = "tile-cache"
DEPTH_CACHE = "depth-cache"
ADAPTER_CACHE = "adapter-cache"
MODEL_ANNOTATOR_CACHE = "annotator-cache"
midas_url = "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-depth-midas-annotator.tar"

def download_and_extract(url: str, dest: str):
    try:
        if os.path.exists("/src/tmp.tar"):
            os.remove("/src/tmp.tar")
        subprocess.check_call(["wget", "-O", "/src/tmp.tar", url])
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(dest)
    tar.close()
    os.remove("/src/tmp.tar")

download_and_extract(midas_url, MODEL_ANNOTATOR_CACHE)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16, cache_dir=TILE_CACHE
)
controlnet.save_pretrained(TILE_CACHE)

depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",  torch_dtype=torch.float16, cache_dir=DEPTH_CACHE
)
depth_controlnet.save_pretrained(DEPTH_CACHE)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")

pipe = DiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16, cache_dir=SD15_WEIGHTS, vae=vae
)
pipe.save_pretrained(SD15_WEIGHTS)

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", cache_dir=INPAINT_WEIGHTS
)
pipe.save_pretrained(INPAINT_WEIGHTS)
