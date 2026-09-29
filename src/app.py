import gradio as gr
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, MusicgenForConditionalGeneration
from PIL import Image
import torch

# We'll only use quantization for Qwen3 as it is the bigger of the two

qwen_quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load models
print("Loading models into memory...")
transcribe_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                                        quantization_config=qwen_quant_config,
                                                        attn_implementation="sdpa",
                                                        device_map="auto")
transcribe_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

gen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
gen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small",
                                                        torch_dtype=torch.float16,
                                                        device_map="auto")

print("Models successfully loaded!")


GENRE_CHOICES = ["Pop", "Bossa Nova", "Rock", "Electronic", "Blues", "Samba"]

# Brazilian music is scarce in MusicGen's dataset, so we need to pass a more accurate description of what it sounds like

GENRE_TO_PROMPT = {
    "Pop": "pop track with catchy melody, bassy drums and synth",
    "Bossa Nova": "bossa nova with nylon-string guitar, soft brushed drums and jazzy chords",
    "Rock": "rock song with distorted guitars and heavy drums",
    "Electronic": "electronic track with driving synth bass and punchy beats",
    "Blues": "blues track with groovy electric guitar, harmonica and shuffle drums",
    "Samba": "samba with pandeiro and cuíca, upbeat Brazilian groove",
}


def transcribe(image: Image.Image) -> str:
    """Extracts the chords and information about the song from an image of sheet music
        using the Qwen3-VL-2B-Instruct model."""

    if image is None:
        return ""

    prompt = "Try to extract time signature and tempo marking from the sheet music. " \
    "Output only this metadata along with the sequence of chords." \
    "If a chord appears in n consecutive instances, write it n-times." \
    "Divide compasses with '|'."

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
    text = transcribe_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the text and image together
    inputs = transcribe_processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(transcribe_model.device)

    # Generate the output
    with torch.no_grad():
        output_ids = transcribe_model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.15)

    # Get the newly generated tokens
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    song_info = transcribe_processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return song_info


def generate_song(song_info: str, genre: str):
    """Generates the backing-track audio with MusicGen-Small.

    Combines the output from our transcriber with the user input.
    Returns (sampling_rate, audio) for gr.Audio.
    """

    # The models needs the transcriber's output to generate the track. 
    # The user can also manually correct the chords in the text box
    if not song_info or not song_info.strip():
        raise gr.Error("Transcribe sheet music first (or type the chords) before generating audio.")

    genre_prompt = GENRE_TO_PROMPT.get(genre, genre)
    full_prompt = f"{genre_prompt}, {song_info.strip()}, high quality, clear harmony. Each compass begins and ends with '|'."

    inputs = gen_processor(
        text=[full_prompt],
        padding=True,
        return_tensors="pt",
    ).to(gen_model.device)

    with torch.no_grad():
        audio_values = gen_model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=768,
        )

    sampling_rate = gen_model.config.audio_encoder.sampling_rate  # 32000 Hz mono
    audio = audio_values[0, 0].cpu().numpy()

    return (sampling_rate, audio)


with gr.Blocks(title="OMR + MusicGen") as demo:
    gr.Markdown("### Optical Music Recognition with Audio Generation")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label="Upload sheet music")
            transcribe_btn = gr.Button("1. Transcribe")
        with gr.Column():
            abc_output = gr.Textbox(label='ABC notation (editable)', lines=6)

    with gr.Row():
        genre_input = gr.Dropdown(choices=GENRE_CHOICES, value="Pop", label="Genre")
        generate_btn = gr.Button("2. Generate audio", variant="primary")

    audio_output = gr.Audio(label="Generated backing track")

    transcribe_btn.click(fn=transcribe, inputs=image_input, outputs=abc_output)
    generate_btn.click(fn=generate_song, inputs=[abc_output, genre_input], outputs=audio_output)

if __name__ == "__main__":
    demo.launch()
