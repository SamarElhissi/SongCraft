from transformers import pipeline
import re

class LyricsGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None  # Cache the model for reuse

    def load_model(self):
        if self.model is None:
            print("Loading lyrics generation model...")
            self.model  = pipeline("text-generation", model=self.model_name)

    def generate_lyrics(self, text):
        self.load_model()
        response = self.model(text, max_length=150, do_sample=True)
        return response[0]['generated_text']
    
    def split_lyrics_into_lines(self, lyrics, max_line_length=50):
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
