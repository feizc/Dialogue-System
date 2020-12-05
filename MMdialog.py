import torch
from torch import nn 
import os

from transformers import GPT2Model, GPT2PreTrainedModel 

class MMdialog(GPT2PreTrainedModel):
    def __init__(self, config):
        super(MMdialog, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.image_off = nn.Linear(1280*7*7, config.n_embd)
        self.image_inverse_off = nn.Linear(config.n_embd, 1280*7*7)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)
    
    def forward(self, input_embs):
        transformer_outputs = self.transformer(inputs_embeds=input_embs)
        return transformer_outputs[0]




