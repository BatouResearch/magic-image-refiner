import torch
import os
from typing import List
import shutil
import time
from cog import BasePredictor, Input, Path
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from controlnet_aux.midas import MidasDetector
import image_util

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
}

SD15_WEIGHTS = "weights"
INPAINT_WEIGHTS = "inpaint-cache"
TILE_CACHE = "tile-cache"
DEPTH_CACHE = "depth-cache"
MODEL_ANNOTATOR_CACHE = "annotator-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        print("Loading pipeline...")
        st = time.time()

        controlnet = [
            ControlNetModel.from_pretrained(
                TILE_CACHE,
                torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                DEPTH_CACHE,
                torch_dtype=torch.float16
            )
        ]

        self.annotator = MidasDetector.from_pretrained(
            MODEL_ANNOTATOR_CACHE, filename="dpt_large_384.pt", model_type="dpt_large"
        ).to("cuda")

        self.img2img_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SD15_WEIGHTS,
            torch_dtype=torch.float16,
            controlnet=controlnet
        ).to("cuda")

        self.inpaint_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            INPAINT_WEIGHTS,
            torch_dtype=torch.float16,
            controlnet=controlnet
        ).to("cuda")

        print("Setup complete in %f" % (time.time() - st))

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for the model",
            default=None
        ),
        image: Path = Input(
            description="Image to refine",
            default=None
        ),
        mask: Path = Input(
            description="When provided, refines some section of the image. Must be the same size as the image",
            default=None
        ),
        resolution: str = Input(
            description="Image resolution",
            default="original",
            choices=["original", "1024", "2048"]
        ),
        resemblance: float = Input(
            description="Conditioning scale for controlnet",
            default=0.75,
            ge=0,
            le=1,
        ),
        creativity: float = Input(
            description="Denoising strength. 1 means total destruction of the original image",
            default=0.25,
            ge=0,
            le=1,
        ),
        hdr: float = Input(
            description="HDR improvement over the original image",
            default=0,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        steps: int = Input(
            description="Steps", default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(
            description="Seed", default=None
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant",
        ),
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
    ) -> List[Path]:
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        loaded_image = self.load_image(image)
        og_resolution = loaded_image.size
        control_image = image_util.resize_for_condition_image(loaded_image, resolution)
        control_depth_image = self.annotator(control_image, detect_resolution=512, image_resolution=control_image.size[1])
        if (resolution == "original"):
            control_depth_image = image_util.resize_for_condition_image(control_depth_image, og_resolution[0])

        final_image = image_util.create_hdr_effect(control_image, hdr)
        
        args = {
            "prompt": prompt,
            "image": final_image,
            "control_image": [final_image, control_depth_image],
            "strength": creativity,
            "controlnet_conditioning_scale": [resemblance, 1],
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": steps,
            "guess_mode": guess_mode,
        }
        pipe = self.img2img_pipe

        if (mask):
            pipe = self.inpaint_pipe
            mask_image = self.load_image(mask)
            args["mask_image"] = mask_image
            if (resolution != "original"):
                raise Exception("Can't upscale and inpaint at the same time")
            if (mask_image.size != loaded_image.size):
                raise Exception("Image and mask must have the same size")
                
        pipe.safety_checker = None
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_sequential_cpu_offload()
        outputs = pipe(**args)
        output_paths = []
        for i, sample in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
