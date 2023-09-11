import torch
import shutil
import os
from tqdm import tqdm
import torchaudio
from glob import glob
import pandas as pd

in_dir = "../AISHELL3"
out_dir = "../AISHELL3_mas"
train_dir = os.path.join(in_dir, "train")
files = glob(os.path.join(train_dir, "**/*.wav"), recursive=True)
df = pd.read_csv(os.path.join(train_dir, "label_train-set.txt"), sep="|", header=None,skiprows=5)

for file in tqdm(files):
    if not os.path.exists(os.path.dirname(file.replace(in_dir, out_dir))):
        os.makedirs(os.path.dirname(file.replace(in_dir, out_dir)))
    shutil.copy(file, file.replace(in_dir, out_dir))
    text = df[df[0]==file.split("/")[-1].replace(".wav", "")][2].values[0]
    text = text.replace("%", "").replace("$", "")
    out_path = file.replace(in_dir, out_dir).replace(".wav", ".txt")
    with open(out_path, "w") as f:
        f.write(text)
