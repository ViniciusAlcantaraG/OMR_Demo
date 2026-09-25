import os
import sys
import gradio as gr
from transformers import GenerationConfig, AutoProcessor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'legato')))
from PIL import Image
from legato.models import LegatoModel
import torch

# Load model
print("Loading model into memory...")
model = LegatoModel.from_pretrained("guangyangmusic/legato-small")
processor = AutoProcessor.from_pretrained("guangyangmusic/legato-small")

# Loading model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).half()
print("Model sucessfully loaded!")

def transcribe(image: Image.Image) -> str:
    """Transforms an image of sheet music into a string in ABC notation.
    We use the model Legato-Small to perform Optical Music Recognition on our image"""

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if "pixel_values" in inputs: 
        inputs["pixel_values"] = inputs["pixel_values"].half()

    generation_config = GenerationConfig(max_length=2028,
                        num_beams=10,
                        repetition_penalty=1.1)

    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)

    abc_notation = processor.batch_decode(output, skip_special_tokens=True)[0]

    return abc_notation

    
demo = gr.Interface(fn=transcribe,
                    inputs=gr.Image(type='pil', label="Upload sheet music"),
                    outputs=gr.Text(label='ABC notation'))

if __name__ == "__main__":
    demo.launch()