from gtts import gTTS

class VocalsGenerator:
    def __init__(self):
        self.model = None  # Cache the model for reuse

    def generate_vocals(self, text, output_filepath):
        # Synthesize speech (this will generate speech, but it could sound like a melody with the right model)
        if text.strip():  # Only generate audio if the line is not empty
            tts = gTTS(text=text, lang='en')
            tts.save(output_filepath)
        else:
            print("Skipping empty line")
