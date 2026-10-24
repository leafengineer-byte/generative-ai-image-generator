import spaces
import torch
import gradio as gr
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

print("Loading Stable Diffusion v1.5...")

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

pipe = pipe.to("cuda")

print("Model loaded successfully!")


@spaces.GPU(duration=120)
def generate_image(
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    seed
):
    if prompt is None or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    try:
        generator = torch.Generator(
            device="cuda"
        ).manual_seed(int(seed))

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator
            )

        return result.images[0]

    except Exception as e:
        raise gr.Error(f"Image generation failed: {e}")


with gr.Blocks(
    title="Generative AI Image Generator"
) as demo:

    gr.Markdown("""
    # 🎨 Generative AI Image Generator
    ### Generate images from natural-language prompts
    This application uses the pre-trained
    **Stable Diffusion v1.5** model.
    """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="📝 Image Prompt",
                placeholder="Describe the image you want...",
                lines=5
            )

            negative_prompt = gr.Textbox(
                label="🚫 Negative Prompt",
                value="blurry, low quality, distorted, deformed, ugly",
                lines=3
            )

            steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=1,
                label="⚙️ Inference Steps"
            )

            guidance_scale = gr.Slider(
                minimum=1,
                maximum=15,
                value=7.5,
                step=0.5,
                label="🎯 Guidance Scale"
            )

            seed = gr.Number(
                value=42,
                precision=0,
                label="🌱 Seed"
            )

            generate_button = gr.Button(
                "🚀 Generate Image",
                variant="primary"
            )

        with gr.Column():
            output_image = gr.Image(
                label="🖼️ Generated Image",
                type="pil"
            )

    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
            seed
        ],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch()
