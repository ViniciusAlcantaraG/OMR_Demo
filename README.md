### Optical Music Recognition with Audio Generation

This projects transforms the picture of sheet music and generates the backing track for the song. For this multimodal task we use $\texttt{Qwen3-VL-2B-Instruct}$ for extracting relevant information from the picture and $\texttt{MusicGen-Small}$ to generate the audio track based on this information.

More specifically:

* $\texttt{Qwen3-VL-2B-Instruct}$ attempts to extract the time signature and the tempo marking along with the song's chords. 
* The user's prompt will guide MusicGen-Small to use this information to reconstruct the song's harmony in whichever genre the user chooses.
* The output is the audio track in TODO format 