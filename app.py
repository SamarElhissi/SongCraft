from audiocraft.models import MusicGen
import os
import streamlit as st
import torch
import numpy as np
import torchaudio
import base64
from transformers import pipeline
import librosa
from gtts import gTTS
import re
from pydub import AudioSegment

st.set_page_config(page_title="MusicGen", page_icon="🎵")

# Setup the logger for TTS (optional, for debugging)
#setup_logger()

# Load a pre-trained model from Coqui TTS (a speech model in this case)
# model_name = "tts_models/en/ljspeech/tacotron2-DDC"
# config = load_config(f'{model_name}/config.json')
# tts = TTS(config)

@st.cache_resource
def load_music_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

@st.cache_resource
def load_lyrics_model():
    model = pipeline("text-generation", model="gpt2")
    return model

def generate_lyrics(prompt):
    generator = load_lyrics_model()
    response = generator(prompt, max_length=150, do_sample=True)
    return response[0]['generated_text']

def split_lyrics_into_lines(lyrics, max_line_length=50):
    # Split by punctuation to get meaningful segments
    raw_lines = re.split(r'[.,;!?]', lyrics)
    lyric_lines = []
    for line in raw_lines:
        line = line.strip()
        if line:  # Skip empty lines
            # Split long lines further into chunks of max_line_length
            while len(line) > max_line_length:
                split_index = line[:max_line_length].rfind(' ')  # Find the last space before max_line_length
                if split_index == -1:  # No space found
                    split_index = max_line_length
                lyric_lines.append(line[:split_index].strip())
                line = line[split_index:].strip()
            if line:
                lyric_lines.append(line.strip())
    return lyric_lines

def generate_music_tensors(text, length):
    model = load_music_model()
    print("Model loaded")
    
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=length)

    output = model.generate(
        descriptions=[text],
        progress=True,
        return_tokens=True
    )

    return output[0]

# Function to synthesize vocals from text
def generate_vocal_audio(lyrics_line, output_filename):
    # Synthesize speech (this will generate speech, but it could sound like a melody with the right model)
    if lyrics_line.strip():  # Only generate audio if the line is not empty
        tts = gTTS(text=lyrics_line, lang='en')
        tts.save("output/" + output_filename)
    else:
        print("Skipping empty line")
    

    
def save_audio(samples:torch.Tensor):
    sample_rate = int(32000)
    save_path = "output/"
    assert samples.dim() == 2 or samples.dim() == 3
    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]
    
    for i, sample in enumerate(samples):
        audio_path = os.path.join(save_path, f"sample_{i}.wav")
        torchaudio.save(
            audio_path,
            sample,
            sample_rate
        )

def get_binary_file_download(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{bin_file}">Download {file_label}</a>'
    return href

def sync_vocals_with_beats(beat_audio_path, lyrics_to_beats, vocal_files_dir):

    if not os.path.exists(beat_audio_path):
        print(f"File not found: {beat_audio_path}")

    # Load the beat audio
    beat_audio = AudioSegment.from_file(beat_audio_path)
    
    # Create an empty audio segment for the final track
    final_audio = AudioSegment.silent(duration=len(beat_audio))
    
    for idx, (lyrics_line, start_time, end_time) in enumerate(lyrics_to_beats):
        vocal_file_path = os.path.join(vocal_files_dir, f"vocal_{idx}.wav")
        
        if os.path.exists(vocal_file_path):
            # Load the vocal audio
            vocal_audio = AudioSegment.from_file(vocal_file_path)
            
            # Determine the start and end times in milliseconds
            start_time_ms = int(start_time * 1000)
            end_time_ms = int(end_time * 1000)
            
            # Adjust vocal duration to match the beat segment
            duration_ms = end_time_ms - start_time_ms
            vocal_audio = vocal_audio[:duration_ms]  # Trim if longer
            if len(vocal_audio) < duration_ms:
                silence = AudioSegment.silent(duration=duration_ms - len(vocal_audio))
                vocal_audio += silence  # Add silence if shorter
            
            # Overlay the vocal audio onto the beat track
            final_audio = final_audio.overlay(vocal_audio, position=start_time_ms)
        else:
            print(f"Vocal file {vocal_file_path} does not exist, skipping.")
    
    return final_audio

def main():
    st.title("Song Generation from text")
    with st.expander("ℹ️ About"):
        st.write("MusicGen is a model that generates music from scratch. It is trained on a large dataset of music and can generate music in various styles and genres.")
    text = st.text_area("Enter some text to generate music")
    slider = st.slider("Length of music to generate", 1, 10, 5)

    if text and slider:
        st.json({"text": text, "length": slider})
        st.subheader("Generate Music")
        music_tensors = generate_music_tensors(text, slider)
        print("Music tensors generated", music_tensors)
        save_file = save_audio(music_tensors)
        audio_file_path = "output/sample_0.wav"
        audio_file = open(audio_file_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        st.markdown(get_binary_file_download(audio_file_path, "Audio"), unsafe_allow_html=True)

        y, sr = librosa.load(audio_file_path, sr=None)  # y is the audio time series, sr is the sample rate

        # Get the tempo (beats per minute) and the beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert beat frames to time (seconds)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        print("Tempo:", tempo)
        print("Beat times (in seconds):", beat_times)

        lyrics = generate_lyrics(text)
        #lyrics ="warm song about love and love and all the different ways things go, all the different ways to live, he added. We've been talking lately about not being there for me, living with my own emotions and trying not to feel like my self is my own. A year ago, while touring with his new band, I'll Be With You, he did an appearance on NPR's Outro and posted a video to YouTube of him talking about the band being with them. The video has since went viral and he's currently tweeting to his fans on Instagram."
        print("Lyrics generated")
        st.json({"Generated Lyrics": lyrics})
        lyrics_lines = split_lyrics_into_lines(lyrics)
        # Ensure we have enough beats for the lyrics
        num_beats = len(beat_times)
        num_lines = len(lyrics_lines)

        # If there are more beats than lyrics lines, we can repeat the lyrics or adjust it
        # If there are more lyrics than beats, we might need to stretch the lyrics across beats
        beats_per_lyric = num_beats // num_lines

        # Map each line to a beat time range
        lyrics_to_beats = []
        for i in range(num_lines):
            start_time = beat_times[i * beats_per_lyric] if i * beats_per_lyric < num_beats else beat_times[-1]
            end_time = beat_times[(i + 1) * beats_per_lyric] if (i + 1) * beats_per_lyric < num_beats else beat_times[-1]
            lyrics_to_beats.append((lyrics_lines[i], start_time, end_time))

        print("Mapped Lyrics to Beats:", lyrics_to_beats)

        # Example of generating vocal audio for each line
        for idx, (line, start_time, end_time) in enumerate(lyrics_to_beats):
            audio_filename = f"vocal_{idx}.wav"
            generate_vocal_audio(line, audio_filename)

        # Example usage:
        final_audio = sync_vocals_with_beats(audio_file_path, lyrics_to_beats, "output")

        # Export the combined audio
        final_audio.export("output/final_track.mp3", format="mp3")
        print("Final track saved as 'final_track.mp3'")

if __name__ == "__main__":
    main()