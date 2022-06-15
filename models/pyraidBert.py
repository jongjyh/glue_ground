from math import pow
import time
from transformers.models.bert.modeling_bert import *
import  torch.nn.functional as F
import numpy as np
import torch
class PyraidBertModel(BertModel):
    def __init__(self,config,add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = PyraidBertEncoder(config)

        # Initialize weights and apply final processing
        self.init_weights()
def get_reverse_set(tsr,seq_len):
    bse = tsr.shape[0]
    full = torch.arange(0,seq_len,dtype=torch.long).repeat((bse,1)).numpy()
    _tsr = tsr.cpu().numpy()
    ret = []
    for b in range(bse):
        _ret = np.setdiff1d(full[b],_tsr[b],True)    
        ret.append(torch.from_numpy(_ret)) 
    ret = torch.stack(ret).to(tsr.device)
    return ret 
class PyraidBertEncoder(BertEncoder):
    def __init__(self,config):
        super().__init__(config)
        self.m_batch = self.config.m_batch if hasattr(self.config,"m_batch") else "k-1"
        self.prune_upto = self.config.prune_upto if hasattr(self.config,"prune_upto")  else 3
        self.prune_rate = pow(self.config.prune_rate,1/self.prune_upto) if  hasattr(self.config,"prune_rate") else pow(0.5,1/self.prune_upto) 
        
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            device = hidden_states.device
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if 0 < i <= self.prune_upto:
                # start = time.time()
                coreset_idx = torch.zeros((hidden_states.shape[0],1),dtype=torch.long).to(device)
                seq_len = hidden_states.shape[1]
                keep_len = int(seq_len*self.prune_rate)
                while coreset_idx.shape[1] < keep_len:
                    seq_arange = torch.arange(0,hidden_states.shape[0]).long().unsqueeze(1).to(device)
                    coreset_hidden_state = hidden_states[seq_arange,coreset_idx]
                    normal_idx = get_reverse_set(coreset_idx,seq_len)
                    normal_hidden_state = hidden_states[seq_arange,normal_idx]
                    
                    cosine_sim =[]
                    # calculate the cos similarity
                    for j in range(normal_idx.shape[1]):
                        _cosine_sim = F.cosine_similarity(coreset_hidden_state,normal_hidden_state[:,j:j+1],dim=2)
                        _cosine_sim = torch.max(_cosine_sim,dim=1)[0]
                        cosine_sim.append(_cosine_sim.unsqueeze(1))
                    
                    cosine_sim = torch.cat(cosine_sim,dim=1)
                    if self.m_batch == 'k-1':
                        m_batch = keep_len -1
                    elif self.m_batch == '1':
                        m_batch = 1
                    selected_center_idx = cosine_sim.topk(m_batch,dim=1,largest=False)[1]
                    # project to origin idx 
                    selected_center_idx = normal_idx[torch.arange(0,selected_center_idx.shape[0]).view(-1,1),selected_center_idx]
                    coreset_idx = torch.cat([coreset_idx,selected_center_idx],dim=1)
                hidden_states = hidden_states[seq_arange,coreset_idx]
                attention_mask = attention_mask.squeeze()[seq_arange,coreset_idx].unsqueeze(dim=1).unsqueeze(dim=1)
                # end = time.time()
                # print(f"{end-start}")

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class PyraidBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = PyraidBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
