from components.music_generator import MusicGenerator
from components.lyrics_generator import LyricsGenerator
from components.vocals_generator import VocalsGenerator
from components.audio_sync import AudioSync

from utils.file_utils import get_binary_file_download
import streamlit as st

st.set_page_config(page_title="MusicGen", page_icon="🎵")

def main():
    
    # Initialize objects
    music_gen = MusicGenerator(model_name="facebook/musicgen-small")
    lyrics_gen = LyricsGenerator(model_name="gpt2")
    vocals_gen = VocalsGenerator()
    audio_sync = AudioSync()

    with st.expander("ℹ️ About"):
        st.write("MusicGen is a model that generates music from scratch. It is trained on a large dataset of music and can generate music in various styles and genres.")
    
    text = st.text_area("Enter some text to generate music")
    slider = st.slider("Length of music to generate", 1, 10, 5)

    if st.button("Generate Song") and text:
        st.json({"text": text, "length": slider})
        st.subheader("Generate Music")

        # Step 1: Generate music
        music_tensor = music_gen.generate_music(text, slider)
        audio_file_path = "output/music.wav"
        music_gen.save_music(music_tensor, audio_file_path)
        
        ## Display audio in the page
        audio_file = open(audio_file_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        st.markdown(get_binary_file_download(audio_file_path, "Audio"), unsafe_allow_html=True)

        # Step 2: Generate lyrics
        lyrics = lyrics_gen.generate_lyrics(text)
        st.json({"Generated Lyrics": lyrics})
        lyrics_lines = lyrics_gen.split_lyrics_into_lines(lyrics)
        
        # Step 3: Map Lyrics to Beats
        lyrics_to_beats = audio_sync.map_lyrics_to_beats(lyrics_lines, audio_file_path)
        
        # Step 4: Generate vocals
        for idx, (line, start_time, end_time) in enumerate(lyrics_to_beats):
            audio_filename = f"output/vocal_{idx}.wav"
            vocals_gen.generate_vocals(line, audio_filename)

        # Step 5: Sync vocals with beats 
        final_audio = audio_sync.sync_vocals_with_beats(audio_file_path, "output", lyrics_to_beats)

        ## Export the combined audio
        final_audio.export("output/final_track.mp3", format="mp3")
        print("Final track saved as 'final_track.mp3'")
        
    else:
        st.warning("Please enter some lyrics.")

if __name__ == "__main__":
    main()
