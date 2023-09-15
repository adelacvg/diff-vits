import dac
from audiotools import AudioSignal
import torchaudio
import librosa
import torch

# Download a model
# model_path = dac.utils.download(model_type="44khz")
model_path = 'weights_44khz_16kbps.pth'
model = dac.DAC.load(model_path)

model.to('cuda')

wave, sr  = torchaudio.load('song.wav')
# wave, sr = librosa.load('song.wav',mono=False)
# wave = torch.as_tensor(wave)
# wave = torchaudio.transforms.Resample(sr,44100)(wave)
# wave = wave.mean(dim=0)
# Load audio signal file
signal = AudioSignal(wave[0,10000000:130000000], sr)#15018932

# Encode audio signal as one long file
# (may run out of GPU memory on long files)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

x = model.compress(signal)

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write('output.wav')