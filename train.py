from bv2 import Trainer

trainer = Trainer(cfg_path='config.json')
# trainer.load('logs/tts/2023-09-11-23-59-58/model-2.pt')
trainer.train()
