import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Image Generator CPU")

MODEL_ID = "SG161222/Realistic_Vision_V5.1"


# ============================================
# LOAD MODEL — CPU-FRIENDLY VERSION
# ============================================
print("Loading model in CPU mode...")

torch.set_grad_enabled(False)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    low_cpu_mem_usage=True
)

# Important: CPU mode
pipe.to("cpu")

# Optional but recommended
try:
    pipe.enable_model_cpu_offload()
except:
    pass

print("Model loaded.")


# ============================================
# API INPUT MODEL
# ============================================
class Prompt(BaseModel):
    prompt: str
    steps: int = 25
    guidance: float = 7.0


# ============================================
# GENERATE ROUTE
# ============================================
@app.post("/generate")
def generate(data: Prompt):
    image = pipe(
        data.prompt,
        num_inference_steps=data.steps,
        guidance_scale=data.guidance
    ).images[0]

    # Convert to base64 for API output
    import io, base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    return {"image_base64": img_base64}
