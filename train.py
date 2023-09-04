from model import Trainer

trainer = Trainer(cfg_path='config.json')
trainer.load('/home/hyc/NS2VC/logs/tts/2023-09-01-21-43-18/model-1.pt')
trainer.train()
