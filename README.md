
# Diff-VITS(WIP)

# VITS with diffusion for zero shot tts.

### Dataset
You should put your dataset in the dataset folder, and the dataset should be organized as follows:

```
dataset
├── p225
│   ├── p225_001.wav
|   ├── P225_001.text
│   ├── p225_002.wav
|   ├── P225_002.text
│   ├── ...
├── p226
│   ├── p226_001.wav
|   ├── P226_001.text
│   ├── p226_002.wav
|   ├── P226_002.text
│   ├── ...
```
and processed dataset will be saved in the processed_dataset folder under the same folder as the dataset folder.

### Data preprocessing

Put the textgird files in the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
```

### Requirements
Use the following command to initialize the env and install the requirements.
```bash
bash init.sh
```
And env named vocos will be created.

### Training
Run `accelerate config` to generate the config file, and then train the model.

Multilingual is in test, now only support mandarin.


```python
accelerate launch train.py
```
### Inference
Run `python tts_infer.py` for inference. You should change the text and model_path in the tts_infer.py file before running the command.

### Q&A
qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
