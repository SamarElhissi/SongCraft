from audiocraft.models import MusicGen
import torchaudio
import torch

class MusicGenerator:
    def __init__(self, model_name: str):
        self.model_path = model_name
        self.model = None  # Cache the model for reuse

    def load_model(self):
        if self.model is None:
            print("Loading music generation model...")
            self.model = MusicGen.get_pretrained(self.model_path)

    def generate_music(self, text, length):
        self.load_model()
        self.model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=length)

        output = self.model.generate(
            descriptions=[text],
            progress=True,
            return_tokens=True)

        return output[0]

    def save_music(self, samples:torch.Tensor, output_path):
        # Save the music tensor to an audio file
        sample_rate = int(32000)
        assert samples.dim() == 2 or samples.dim() == 3
        samples = samples.detach().cpu()

        if samples.dim() == 2:
            samples = samples[None, ...]
    
        for i, sample in enumerate(samples):
            audio_path = output_path
            torchaudio.save(
                audio_path,
                sample,
                sample_rate)
