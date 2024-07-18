import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Define the image generation function
def generate_image(prompt, width=512, height=512, num_inference_steps=25):
    with torch.no_grad():
        result = pipe(prompt, width=width, height=height, num_inference_steps=num_inference_steps)
        image = result.images[0]
    return image

# Example usage
if __name__ == "__main__":
    prompt = "a red apple on a wooden table"
    image = generate_image(prompt)
    image.show()  # Display the generated image
    image.save("generated_image.png")  # Save the generated image
