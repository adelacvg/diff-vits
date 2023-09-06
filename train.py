from model import Trainer

trainer = Trainer(cfg_path='config.json')
# trainer.load('')
trainer.train()
