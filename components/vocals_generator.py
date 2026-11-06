import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import json
import yaml
import requests
import zipfile
from diffsinger.inference.ds_acoustic import DiffSingerAcousticInfer
from diffsinger.utils.phoneme_utils import load_phoneme_dictionary
from diffsinger.utils.hparams import set_hparams, hparams
from diffsinger.utils.utils import load_ckpt
from diffsinger.modules.toplevel import DiffSingerAcoustic

class VocalsGenerator:
    def __init__(self):
        # Load configuration
        config_path = os.path.join('diffsinger', 'configs', 'base.yaml')
        set_hparams(config=config_path, exp_name='diffsinger')
        
        # Update hparams to point to the correct dictionary
        hparams['dictionary'] = os.path.join('diffsinger', 'dictionaries', 'opencpop-extension.txt')
        hparams['work_dir'] = 'diffsinger'
        
        # Ensure checkpoint directory exists
        ckpt_dir = os.path.join('checkpoints', 'diffsinger')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Download and setup NSF-HiFiGAN vocoder
        self._setup_vocoder()
        
        # Load phoneme dictionary
        self.phoneme_dict = load_phoneme_dictionary()
        
        # Initialize model without loading checkpoint
        self.model = DiffSingerAcousticInfer(load_model=False, load_vocoder=True)
        
        # Create a new model instance without loading checkpoint
        self.model.model = DiffSingerAcoustic(
            vocab_size=len(self.phoneme_dict),
            out_dims=hparams['audio_num_mel_bins']
        ).eval()
        
        # Set up language mapping
        self.model.lang_map = {'en': 0}  # English is language 0
        
        # Set the phoneme dictionary on the model
        self.model.phoneme_dictionary = self.phoneme_dict
        
        # Load the pinyin to phoneme mapping
        self.pinyin_to_phoneme = {}
        with open(os.path.join('diffsinger', 'dictionaries', 'opencpop-extension.txt'), 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                pinyin, phonemes = line.strip().split('\t')
                self.pinyin_to_phoneme[pinyin] = phonemes
    
    def _setup_vocoder(self):
        vocoder_dir = 'checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02'
        vocoder_path = os.path.join(vocoder_dir, 'model.ckpt')
        
        if not os.path.exists(vocoder_path):
            print("Downloading NSF-HiFiGAN vocoder...")
            os.makedirs(vocoder_dir, exist_ok=True)
            
            # Download the vocoder model
            url = "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
            response = requests.get(url, stream=True)
            
            # Save the zip file
            zip_path = os.path.join(vocoder_dir, 'vocoder.zip')
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(vocoder_dir)
            
            # Clean up the zip file
            os.remove(zip_path)
            
            print("NSF-HiFiGAN vocoder downloaded and extracted successfully.")
    
    def _convert_to_phonemes(self, text):
        # Convert text to lowercase and split into words
        words = text.lower().split()
        phoneme_words = []
        
        for word in words:
            # Try to find the word in the pinyin dictionary
            if word in self.pinyin_to_phoneme:
                phoneme_words.append(self.pinyin_to_phoneme[word])
            else:
                # If word not found, try to split it into individual characters
                # and map each character to its phoneme
                phonemes = []
                for char in word:
                    if char in self.pinyin_to_phoneme:
                        phonemes.append(self.pinyin_to_phoneme[char])
                    else:
                        # If character not found, use a default phoneme
                        phonemes.append('SP')  # SP for silence/space
                phoneme_words.append(' '.join(phonemes))
        
        # Join all phonemes with spaces and ensure proper format
        phoneme_seq = ' '.join(phoneme_words)
        
        # Verify that all phonemes are in the dictionary
        for phoneme in phoneme_seq.split():
            if phoneme not in self.phoneme_dict._phone_to_id:
                print(f"Warning: Phoneme '{phoneme}' not found in dictionary")
        
        return phoneme_seq
    
    def generate_vocals(self, lyrics, output_path):
        # Convert lyrics to phonemes
        phoneme_seq = self._convert_to_phonemes(lyrics)
        words = phoneme_seq.split()
        
        # Create a DS file for input
        ds_file = {
            "name": "generated_song",
            "ph_seq": phoneme_seq,  # Use the converted phoneme sequence
            "note_seq": [60] * len(words),  # Default to middle C
            "note_dur": [0.5] * len(words),  # Default duration
            "is_slur_seq": [0] * len(words),
            "ph_dur": [0.5] * len(words),
            "ph_num": len(words),
            "frames": int(len(words) * hparams['hop_size']),
            "mel": None,
            "f0": None,
            "energy": None,
            "breathiness": None,
            "voicing": None,
            "tension": None,
            "lang": "en"
        }
        
        # Save DS file
        ds_path = os.path.join('temp', 'input.ds')
        os.makedirs('temp', exist_ok=True)
        with open(ds_path, 'w') as f:
            json.dump(ds_file, f)
        
        # Generate audio using the DS file
        audio = self.model.run_inference([ds_file])
        
        # Save output
        sf.write(output_path, audio, hparams['audio_sample_rate'])
        
        # Clean up
        os.remove(ds_path)
