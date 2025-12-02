import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"

print("Loading Realistic Vision… (first time takes a few minutes)")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16" if torch.cuda.is_available() else None,
    safety_checker=None       # désactiver le filtre NSFW si tu le veux
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

app = FastAPI(
    title="Realistic Vision Image Generator API",
    description="Génération d'images réalistes avec Realistic Vision 5.1",
)

class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, watermark"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    seed: int | None = None

@app.post("/generate")
def generate(req: GenRequest):
    try:
        generator = torch.Generator(device=device)
        if req.seed is not None:
            generator = generator.manual_seed(req.seed)

        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        )

        image = result.images[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "model": MODEL_ID,
        "prompt": req.prompt,
        "image_base64": b64
    })
