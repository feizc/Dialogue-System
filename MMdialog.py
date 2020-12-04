import torch
from torch import nn 
import os

from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config, BertTokenizer

class MMdialog(GPT2PreTrainedModel):
    def __init__(self, config):
        super(MMdialog, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)
    
    def forward(self, input_embs):
        transformer_outputs = self.transformer(inputs_embeds=input_embs)
        return transformer_outputs[0]




