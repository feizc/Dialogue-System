import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss  
import os

from transformers import * 

emotion_num = 62 


class LabelPredict(GPT2PreTrainedModel):
    
    def __init__(self, config):
        super(LabelPredict, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.label_classifier = nn.Linear(config.n_embd, 274)

        self.init_weights() 

    def forward(self, his, respond):
        transformer_outputs = self.transformer(his)
        hidden_states = transformer_outputs[0][:, -1, :] 
        # print(hidden_states.size())
        logits = self.label_classifier(hidden_states)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, respond)
        return loss, logits  


class LabelBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 274 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, his, respond):
        outputs = self.bert(input_ids=his)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), respond.view(-1))
        return loss, logits



class MMdialog(GPT2PreTrainedModel): 
    # mode: dialog / emotion
    def __init__(self, config):
        super(MMdialog, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.image_off = nn.Linear(1280*7*7, config.n_embd)
        self.image_inverse_off = nn.Linear(config.n_embd, 1280*7*7)
        self.emotion_classifier = nn.Linear(config.n_embd, emotion_num)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)
    
    def inference(self, input_embs, token_type_ids):
        transformer_outputs = self.transformer(inputs_embeds=input_embs, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def forward(self, input_embs, token_type_ids, labels, image_feature, mode='dialog'):

        transformer_outputs = self.transformer(inputs_embeds=input_embs, token_type_ids=token_type_ids)
        # outputs = (hidden_states, presents, all_hidden_states, all_self_attentions)
        hidden_states = transformer_outputs[0]
        txt_hidden_states, img_hidden_states = hidden_states[:-1, :], hidden_states[-1, :].unsqueeze(0) 

        if mode == 'emotion': 
            emotion_logits = self.emotion_classifier(img_hidden_states) 
            emo_loss_fct = CrossEntropyLoss(ignore_index=-100)
            emo_loss = emo_loss_fct(emotion_logits, labels[-1].view(-1))
            return emo_loss 

        if txt_hidden_states.size(0) > 1:
            lm_logits = self.lm_head(txt_hidden_states)
            txt_loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_txt = txt_loss_fct(lm_logits, labels)
            loss = loss_txt
        # print(image_feature)
        # image_tag = torch.zeros((1, input_embs.size(-1))
        if image_feature[0][0] != 0.:
            #img_regs = self.image_inverse_off(img_hidden_states)
            img_regs = img_hidden_states
            #print(img_regs.size())
            #print(image_feature.size())
            img_loss_fct = MSELoss()
            loss_img = img_loss_fct(img_regs, image_feature)
            loss += loss_img

        return lm_logits, loss, img_hidden_states  


class VEID(GPT2PreTrainedModel):

    def __init__(self, config): 
        super(VEID, self).__init__(config) 
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        self.image_inverse = nn.Linear(config.n_embd, 512) 

        self.init_weights() 
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 

    def img_loss(self, img, target, img_bank): 
        img = img / torch.norm(img, 1) 
        target = target / torch.norm(target, 1) 
        fenzi = img * target 
        fenzi = torch.sum(fenzi) 
        fenzi = torch.exp(fenzi)
        #print(fenzi.size())
        fenmu = img * img_bank 
        fenmu = torch.sum(fenmu, dim=1)
        fenmu1 = torch.exp(fenmu)
        fenmu1 = torch.sum(fenmu1) 
        return fenzi / fenmu1, fenmu
        


    def forward(self, input_ids, token_type_ids, target_ids, target_feature, img_bank):
        transformer_outputs = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0] 
        
        logits = self.lm_head(hidden_states)
        txt_loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss_txt = txt_loss_fct(logits.squeeze(0), target_ids.squeeze(0)) 

        img_feature = hidden_states[:, -1, :]
        img_feature = self.image_inverse(img_feature) 
        img_loss, fenmu = self.img_loss(img_feature, target_feature, img_bank)

        return loss_txt+1000*img_loss, loss_txt, 1000*img_loss, fenmu  
    
