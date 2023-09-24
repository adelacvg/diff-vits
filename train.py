from model3 import Trainer

trainer = Trainer(cfg_path='config.json')
# trainer.load('logs/tts/2023-09-18-02-10-02/model-108.pt')
trainer.train()
