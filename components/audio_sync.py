from pydub import AudioSegment
import librosa
import os

class AudioSync:
    def __init__(self):
        self.model = None  # Cache the model for reuse

    def map_lyrics_to_beats(self, lyrics_lines, audio_file_path):
        y, sr = librosa.load(audio_file_path, sr=None)  # y is the audio time series, sr is the sample rate

        # Get the tempo (beats per minute) and the beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert beat frames to time (seconds)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

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
