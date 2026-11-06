from components.music_generator import MusicGenerator
from components.lyrics_generator import LyricsGenerator
from components.vocals_generator import VocalsGenerator
from components.audio_sync import AudioSync
import torch
import streamlit as st
import os
import tempfile

from utils.file_utils import get_binary_file_download

st.set_page_config(page_title="SongCraft", page_icon="🎵")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Initialize objects
    music_gen = MusicGenerator(model_name="facebook/musicgen-medium")
    lyrics_gen = LyricsGenerator(model_name="gpt-4o", device=device)
    vocals_gen = VocalsGenerator()
    audio_sync = AudioSync()

    # Create temporary directory for audio files
    temp_dir = tempfile.mkdtemp()

    # Add voice upload section
    st.header("Voice Reference (Optional)")
    reference_voice = st.file_uploader("Upload a reference voice recording (WAV format)", type=['wav'])
    if reference_voice:
        reference_path = os.path.join(temp_dir, "reference_voice.wav")
        with open(reference_path, "wb") as f:
            f.write(reference_voice.getvalue())
        vocals_gen = VocalsGenerator(reference_path)

    with st.expander("ℹ️ About"):
        st.write("SongCraft is an AI-powered music generator that creates complete songs with lyrics and music. It uses advanced models to generate realistic singing and musical accompaniment.")
    
    text = st.text_area("Enter a theme or topic for your song")
    slider = st.slider("Length of music to generate (in seconds)", 30, 300, 120)
    pitch_shift = st.slider("Vocal pitch adjustment (in semitones)", -12, 12, 0)

    if st.button("Generate Song") and text:
        st.subheader("Generating your song...")
        
        # Step 1: Generate music
       # music_tensor = music_gen.generate_music(text, slider)
        audio_file_path = "output/music.wav"
       # music_gen.save_music(music_tensor, audio_file_path)
        
        # Display the instrumental track
        st.subheader("Instrumental Track")
        audio_file = open(audio_file_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        st.markdown(get_binary_file_download(audio_file_path, "Download Instrumental"), unsafe_allow_html=True)

        # Step 2: Generate lyrics
        #lyrics = lyrics_gen.generate_lyrics(text)
        lyrics = """I remember the night when the stars burned bright,
Your hand in mine felt so damn right.
We laughed in the glow of the soft moon's light,
Never saw the storm creeping up in sight.

Your words were soft, like a summer breeze,
Promised me love, brought me to my knees.
But the winds changed tune, hearts don't come with guarantees,
Now I'm picking up the pieces of shattered memories.
"""
        st.subheader("Generated Lyrics")
        st.write(lyrics)
        lyrics_lines = lyrics_gen.split_lyrics_into_lines(lyrics)
        
        # Step 3: Map Lyrics to Beats
        lyrics_to_beats = audio_sync.map_lyrics_to_beats(lyrics_lines, audio_file_path)
        
        # Step 4: Generate vocals
        for idx, (line, start_time, end_time) in enumerate(lyrics_to_beats):
            audio_filename = f"output/vocal_{idx}.wav"
            vocals_gen.generate_vocals("ni hao", audio_filename) 

            #vocals_gen.generate_vocals(line, audio_filename)

        # Step 5: Sync vocals with beats 
        final_audio = audio_sync.sync_vocals_with_beats(audio_file_path, lyrics_to_beats, "output")

        # Export the combined audio
        final_audio.export("output/final_track.mp3", format="mp3")
        
        # Display the final result
        st.subheader("Your Complete Song")
        audio_file = open("output/final_track.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
        st.markdown(get_binary_file_download("output/final_track.mp3", "Download Complete Song"), unsafe_allow_html=True)
        
    elif not text:
        st.warning("Please enter a theme or topic for your song.")

if __name__ == "__main__":
    main()
