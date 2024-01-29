from diffusers import ControlNetModel, DiffusionPipeline, AutoencoderKL, AutoPipelineForInpainting
import torch
from RealESRGAN import RealESRGAN


for scale in [2, 4]:
    model = RealESRGAN("cuda", scale=scale)
    model.load_weights(f"weights/RealESRGAN_x{scale}.pth", download=True)

SD15_WEIGHTS = "weights"
INPAINT_WEIGHTS = "inpaint-cache"
TILE_CACHE = "tile-cache"
DEPTH_CACHE = "depth-cache"
ADAPTER_CACHE = "adapter-cache"

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
