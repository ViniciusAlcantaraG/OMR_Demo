import gradio as gr
from transformers import GenerationConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, AutoModelForTextToWaveform
from PIL import Image
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model
print("Loading model into memory...")
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                                        quantization_config=quant_config,
                                                        attn_implementation="sdpa",
                                                        device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

print("Model sucessfully loaded!")

def transcribe(image: Image.Image) -> str:
    """Extracts the chords and information about the song from an image of sheet music 
        using the Qwen3-VL-2B-Instruct model."""

    prompt = "Extract time signature and tempo marking from the sheet music. Output this metadata along with the sequence of chords."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # We use the chat template to interact with the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process the text and image together
    inputs = processor(
        text=[text], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")
    
    # Generate the output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.15)
        
    # Get the newly generated tokens
    generated_ids = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    
    song_info = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    return song_info

#TODO
#def generate_song(song_info: str):



    
demo = gr.Interface(fn=transcribe,
                    inputs=gr.Image(type='pil', label="Upload sheet music"),
                    outputs=gr.Text(label='ABC notation'))

if __name__ == "__main__":
    demo.launch()