from diffusers import ControlNetModel, DiffusionPipeline, AutoencoderKL, AutoPipelineForInpainting
import torch
from RealESRGAN import RealESRGAN


for scale in [2, 4]:
    model = RealESRGAN("cuda", scale=scale)
    model.load_weights(f"weights/RealESRGAN_x{scale}.pth", download=True)

SD15_WEIGHTS = "weights"
INPAINT_WEIGHTS = "inpaint-cache"
CONTROLNET_CACHE = "controlnet-cache"
ADAPTER_CACHE = "adapter-cache"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16, cache_dir=CONTROLNET_CACHE
)
controlnet.save_pretrained(CONTROLNET_CACHE)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")

pipe = DiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16, cache_dir=SD15_WEIGHTS, vae=vae
)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", cache_dir=ADAPTER_CACHE)
pipe.save_pretrained(SD15_WEIGHTS)

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", cache_dir=INPAINT_WEIGHTS
)
pipe.save_pretrained(INPAINT_WEIGHTS)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", cache_dir=ADAPTER_CACHE)
