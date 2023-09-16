import dac
from audiotools import AudioSignal
import torchaudio
import librosa
import torch

# Download a model
# model_path = dac.utils.download(model_type="44khz")
model_path = 'weights_44khz_16kbps.pth'
model = dac.DAC.load(model_path)

model.to('cuda:6')

wave, sr  = torchaudio.load('song.wav')
# wave, sr = librosa.load('song.wav',mono=False)
# wave = torch.as_tensor(wave)
wave = torchaudio.transforms.Resample(sr,44100)(wave).to(model.device)
# wave = wave.mean(dim=0)
# Load audio signal file
signal = AudioSignal(wave[0,10000000:10100000], 44100)#15018932

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode audio signal
y = model.decode(z).squeeze(0)

# Write to file
torchaudio.save('output.wav',y.cpu(),44100)
# y.write('output.wav')