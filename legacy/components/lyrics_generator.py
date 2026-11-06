import re
from openai import OpenAI

class LyricsGenerator:
    def __init__(self, model_name: str, device):
        self.model_name = model_name
        self.model = None  # Cache the model for reuse
        self.device = device

    def load_model(self):
        if self.model is None:
            print("Loading lyrics generation model...")

    def clean_lyrics(self, lyrics):
        # Remove structural markers like "Verse 1:", "Chorus:", etc.
        lyrics = re.sub(r'(Verse|Chorus|Bridge|Pre-Chorus|Intro|Outro)\s*\d*:?\s*', '', lyrics)
        # Remove any remaining numbers or special markers
        lyrics = re.sub(r'^\d+\.\s*', '', lyrics, flags=re.MULTILINE)
        # Remove empty lines
        lyrics = '\n'.join(line for line in lyrics.split('\n') if line.strip())
        return lyrics

    def generate_lyrics(self, text):
        self.load_model()
        client = OpenAI(base_url="https://models.inference.ai.azure.com",
                        api_key="key")

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a song lyrics generator. Write lyrics without any structural markers like 'Verse', 'Chorus', etc. Just write the actual lyrics."},
                {
                    "role": "user",
                    "content": f"Write song lyrics about {text}. Make it rhyme and have a good rhythm. Don't include any structural markers like 'Verse' or 'Chorus'."
                }
            ]
        )
        lyrics = completion.choices[0].message.content
        return self.clean_lyrics(lyrics)
    
    def split_lyrics_into_lines(self, lyrics, max_line_length=50):
        # Split by newlines first
        raw_lines = lyrics.split('\n')
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
