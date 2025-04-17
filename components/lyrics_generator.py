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
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.model  = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
           

    def generate_lyrics(self, text):
        self.load_model()
        # Create prompt
        # prompt = (f"Write a poetic song about love. Include verses, a catchy chorus, "
        #       "and a heartfelt bridge. Use rhymes and emotional depth. Here’s an example of a song:\n\n"
        #       "Verse: The stars above, they shine so bright,\n"
        #       "Chorus: In your arms, I find my home,\n\n"
        #       f"Now write a new song about love:\n\n")    
        # # Encode the input
        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    
        # # Generate lyrics
        # output = self.model.generate(
        #     input_ids,
        #     max_length=150,
        #     temperature=0,
        #     top_p=0.9,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
    
        # # Decode and return the lyrics
        # lyrics = self.tokenizer.decode(output[0], skip_special_tokens=True)
        client = OpenAI(base_url="https://models.inference.ai.azure.com",
                        api_key="key")

        completion = client.chat.completions.create(
        model=self.model_name,
        messages=[
        {"role": "system", "content": "You are a song lyrics generator."},
        {
            "role": "user",
            "content": "Write a song lyric about " + text
        }
        ]
        )
        return completion.choices[0].message.content
    
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
