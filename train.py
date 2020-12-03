import model
from transformers import GPT2Config

config = GPT2Config(vocab_size=13088)
print(config)