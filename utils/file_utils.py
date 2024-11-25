import base64
import os
import torchaudio

def save_audio(samples, path="output/sample.wav", sample_rate=32000):
    torchaudio.save(path, samples.cpu(), sample_rate)

def get_binary_file_download(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{bin_file}">Download {file_label}</a>'
    return href
