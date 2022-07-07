from turtle import forward
import math
from git import Object
from typing import Optional,Tuple
import torch
import torch.nn as nn
import re
from transformers.activations import ACT2FN
from models.search_cells import NAS201SearchCell as SearchCell
from transformers.models.bert.modeling_bert import BertSelfAttention
from models.operation import OPS
# From https://github.com/rabeehk/compacter

ACT2FN["identity"] = lambda x: x
op_names=(
    "none",
    # "avg_pool_3",
    # "max_pool_3",
    # "nor_conv_3",
    # "nor_conv_5",
    # "nor_conv_7",
    # "dil_conv_3",
    # "dil_conv_5",
    # "dil_conv_7",
    # "skip_connect",
    "lora",
)
FORWARD_ARCH={
    "random":"forward_urs",
    "gdas":"forward_gdas",
}

def get_adapter(adapter_type):
    if adapter_type == "normal":
        return NASAdapter
    else:
        raise ValueError("Not Implemented")
class NASAdapter(nn.Module):
    def __init__(self, config, transformer_config,max_node=2,op_names=op_names):
        super().__init__()
        self.adapter_input_size = transformer_config.hidden_size
        # self.non_linearity = ACT2FN[config.adapter_non_linearity]
        self.residual = config.ans_normal_adapter_residual
        self.ans_forward = "gdas" 
        self.ans_forward = FORWARD_ARCH[self.ans_forward]
        self.cells = SearchCell(
                                transformer_config.hidden_size,
                                transformer_config.hidden_size,
                                stride=1,
                                max_nodes=max_node,
                                op_names=op_names,
                                )
        self.num_edges = self.cells.num_edges
        self.op_names = op_names
        self.arch_parameters = torch.nn.Parameter(1e-3 * torch.randn(self.num_edges,len(self.op_names)))
        self.tem = 10.0
        self.register_buffer("tem_proportion",torch.tensor([0.,]))
        self.index = None
        self.search =True


    def forward(self, x):
        def get_gumbel_prob(xins):
            if not self.search:
                return self.hardwts.detach(), self.index.detach()
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbels) / 0.5
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break
            self.tem -= self.tem_proportion.item()   
            self.hardwts, self.index = hardwts.detach(),index.detach()  
            return hardwts, index
        hardwts, index = get_gumbel_prob(self.arch_parameters)
        if self.residual:
            output = getattr(self.cells,self.ans_forward)(x,hardwts,index) + x
        else:
            output = getattr(self.cells,self.ans_forward)(x,hardwts,index) 
        return output


class NASBertFF(nn.Module):
    def __init__(self, bertFF, config, transformer_config,):
        super().__init__()
        self.dense = bertFF.dense
        self.LayerNorm = bertFF.LayerNorm
        self.dropout = bertFF.dropout
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.pattern = config.pattern
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states) 
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class NASBertAttn(BertSelfAttention):
    def __init__(self,bertAttn, config, position_embedding_type=None,transformer_config=None):
        super().__init__(config, position_embedding_type)
        self.num_attention_heads =bertAttn.num_attention_heads  
        self.attention_head_size =bertAttn.attention_head_size  
        self.all_head_size =bertAttn.all_head_size  

        self.query =bertAttn.query  
        self.key =bertAttn.key  
        self.value =bertAttn.value  
        

        self.dropout =bertAttn.dropout      
        self.position_embedding_type =bertAttn.position_embedding_type 
        # self.max_position_embeddings =bertAttn.max_position_embeddings  
        # self.distance_embedding =bertAttn.distance_embedding  

        self.is_decoder =bertAttn.is_decoder  
        self.lora_query = get_adapter(config.ans_adapter_type)(config, transformer_config)
        self.lora_value = get_adapter(config.ans_adapter_type)(config, transformer_config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) + self.lora_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states) + self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

        
    
    
def modify_with_adapters(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*layer.[0-9]*", m_name):
            if config.ans_pattern == "adapter":
                module.output = NASBertFF(
                    module.output,
                    config,
                    transformer.config,
                )
            elif config.ans_pattern == "attention":
                module.attention.self = NASBertAttn(
                    module.attention.self,
                    transformer.config,
                    transformer_config = config,
                ) 
        
    return transformer


if __name__ == "__main__":
    # from transformers import AutoModel
    # from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-t",
    #     "--adapter_type",
    #     required=True,
    #     type=str,
    #     choices=["normal", "lowrank", "compacter"],
    # )
    # args = parser.parse_args()

    # class AdapterConfig:
    #     def __init__(self, adapter_type):
    #         self.adapter_type = adapter_type

    #         if self.adapter_type == "normal":
    #             # Adapter Config
    #             self.adapter_reduction_factor = 16
    #             self.adapter_non_linearity = "relu"
    #             self.normal_adapter_residual = True
    #             self.add_compacter_in_attention = True
    #         elif self.adapter_type == "compacter":
    #             # Compacter
    #             self.adapter_reduction_factor = 16
    #             self.adapter_non_linearity = "relu"
    #             self.compacter_hypercomplex_division = 4
    #             self.compacter_learn_phm = True
    #             self.compacter_hypercomplex_nonlinearity = "xyz"
    #             self.compacter_shared_phm_rule = False
    #             self.compacter_factorized_phm = True
    #             self.compacter_shared_W_phm = False
    #             self.compacter_factorized_phm_rule = False
    #             self.compacter_phm_c_init = "xyz"
    #             self.compacter_phm_rank = 1
    #             self.compacter_phm_init_range = 0.0001
    #             self.compacter_kronecker_prod = False
    #             self.compacter_adapter_non_linearity = "gelu_new"
    #             self.compacter_add_compacter_in_attention = True

    #         self.trainable_param_names = ".*layer_norm.*|.*adapter.*"

    # config = AdapterConfig(args.adapter_type)
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # input_seq = tokenizer(
    #     ["Applies a linear transformation to the incoming data."],
    #     return_tensors="pt",
    # )
    # target_seq = tokenizer(
    #     ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
    #     return_tensors="pt",
    # )

    # print("Old model")
    # # print(model)
    # # print(model.state_dict().keys())
    # old_param = model.state_dict()
    # with torch.no_grad():
    #     old_outputs = model(
    #         input_ids=input_seq.input_ids,
    #         decoder_input_ids=target_seq.input_ids[:, :-1],
    #         labels=target_seq.input_ids[:, 1:],
    #     )

    # model = modify_with_adapters(model, config)
    # new_param = model.state_dict()
    # """
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])
    # """
    # # print(old_param - new_param)
    # print("New model")
    # # print(model)
    # with torch.no_grad():
    #     new_outputs = model(
    #         input_ids=input_seq.input_ids,
    #         decoder_input_ids=target_seq.input_ids[:, :-1],
    #         labels=target_seq.input_ids[:, 1:],
    #     )

    # print("Trainable parameters")
    # """
    # print(
    #     [
    #         p_name
    #         for p_name in dict(model.named_parameters()).keys()
    #         if re.fullmatch(config.trainable_param_names, p_name)
    #     ]
    # )
    # """
    # print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    # print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained("bert-base-uncased").encoder
    config = Object
    config.adapter_type = "normal"
    config.normal_adapter_residual=True
    model = modify_with_adapters(model,config=config) 

    input = torch.randn((10,32,768))
    output = model(input)
    
