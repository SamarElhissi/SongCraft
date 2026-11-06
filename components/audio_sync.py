from pydub import AudioSegment
import librosa
import os
import numpy as np

class AudioSync:
    def __init__(self):
        self.model = None  # Cache the model for reuse

    def map_lyrics_to_beats(self, lyrics_lines, audio_file_path):
        y, sr = librosa.load(audio_file_path, sr=None)

        # Get the tempo (beats per minute) and the beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert beat frames to time (seconds)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Calculate the total duration of the song
        total_duration = len(y) / sr

        # Calculate average duration per line based on number of words
        total_words = sum(len(line.split()) for line in lyrics_lines)
        avg_duration_per_word = total_duration / total_words

        # Map each line to a time range based on word count
        lyrics_to_beats = []
        current_time = 0
        for line in lyrics_lines:
            word_count = len(line.split())
            line_duration = word_count * avg_duration_per_word
            
            # Ensure we don't exceed total duration
            if current_time + line_duration > total_duration:
                line_duration = total_duration - current_time
            
            end_time = current_time + line_duration
            lyrics_to_beats.append((line, current_time, end_time))
            current_time = end_time

        return lyrics_to_beats
    
    def sync_vocals_with_beats(self, beat_audio_path, lyrics_to_beats, vocal_files_dir):
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
            
                # Adjust vocal duration to match the time segment
                duration_ms = end_time_ms - start_time_ms
                vocal_audio = vocal_audio[:duration_ms]  # Trim if longer
                
                # Apply fade in/out for smoother transitions
                fade_duration = min(200, duration_ms // 4)  # Longer fade for smoother transitions
                vocal_audio = vocal_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                # Apply volume adjustment to blend with music
                vocal_audio = vocal_audio - 3  # Reduce volume by 3dB to blend better with music
                
                if len(vocal_audio) < duration_ms:
                    silence = AudioSegment.silent(duration=duration_ms - len(vocal_audio))
                    vocal_audio += silence  # Add silence if shorter
            
                # Overlay the vocal audio onto the beat track
                final_audio = final_audio.overlay(vocal_audio, position=start_time_ms)
            else:
                print(f"Vocal file {vocal_file_path} does not exist, skipping.")
    
        # Mix the vocals with the original music
        final_audio = beat_audio.overlay(final_audio)
        
        return final_audio
