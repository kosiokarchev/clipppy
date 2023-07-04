from warnings import filterwarnings

filterwarnings('ignore', module='torch.nn.modules.lazy')
filterwarnings('ignore', message='The dataloader, .*, does not have many workers')
filterwarnings('ignore', message='Named tensors')
filterwarnings('ignore', module='seaborn.distributions', message='elementwise comparison failed')
