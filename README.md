### Optical Music Recognition with Audio Generation

This projects transforms the picture of sheet music and generates the backing track for the song. For this multimodal task we use $\texttt{Qwen3-VL-2B-Instruct}$ for extracting relevant information from the picture and $\texttt{MusicGen-Small}$ to generate the audio track based on this information.

More specifically:

* $\texttt{Qwen3-VL-2B-Instruct}$ attempts to extract the time signature and the tempo marking along with the song's chords. 
* The user's prompt will guide MusicGen-Small to use this information to reconstruct the song's harmony in whichever genre the user chooses.
  1. Upload sheet music and click **Transcribe** to get time signature, tempo and chords (editable ABC text).
  2. Pick a genre from **Pop, Bossa Nova, Rock, Electronic, Blues, Samba** and click **Generate audio**.
* MusicGen-Small is loaded in `float16` (no bitsandbytes quantization: it is only ~300M params, and 8-bit is slower and can degrade EnCodec audio quality — quantization is kept only for Qwen3-VL-2B).
* The output is the audio track as 32 kHz mono audio in the Gradio player (~15s by default, `max_new_tokens=768`; use 1024 for ~20s). 