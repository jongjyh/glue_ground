from turtle import forward
from  transformers.models.bert.modeling_bert import *
import torch.nn as nn
class LadderLayers(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.r = config.r if hasattr(config,'r') else 8
        self.intermediate_dim = config.intermediate_size//self.r
        self.down = nn.Linear(config.intermediate_size,self.intermediate_dim)
        self.intermediate = nn.Linear(self.intermediate_dim,self.intermediate_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        # self.temperature = config.temperature if hasattr(config,"temperature") else 0.1
        self.temperature = 0.1
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self,backbone_hidden_states,ladder_hidden_states):
        # mixing the input by gate variable
        mu = torch.sigmoid(self.gate/self.temperature)
        down_hidden_states = self.down(backbone_hidden_states)
        inputs = mu * down_hidden_states + (1-mu) * ladder_hidden_states
        # inputs = self.intermediate_act_fn(inputs)
        outputs_states = self.intermediate(inputs)
        outputs_states = self.intermediate_act_fn(outputs_states)
        return outputs_states
        
class Ladder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.num_layers = config.num_layers if hasattr(config,'num_layers') else config.num_hidden_layers
        self.ladder_layers = nn.ModuleList([LadderLayers(config) for _ in range(self.num_layers)])
        self.r = config.r if hasattr(config,'r') else 8 
        self.intermediate_dim = config.intermediate_size//self.r
        self.down = nn.Linear(config.hidden_size,self.intermediate_dim)
        self.up = nn.Linear(self.intermediate_dim,config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def init_ladder_states(self,hidden_states):
        # return self.intermediate_act_fn(self.down(hidden_states))
        return self.down(hidden_states)
    
    def output_last_logits(self,hidden_states):
        # return self.intermediate_act_fn(self.up(hidden_states))
        return self.up(hidden_states)

        
class LadderBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([LadderBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.ladder = Ladder(config)
        
        

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True) -> Union[Tuple[torch.Tensor],BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
    
        ladder_hidden_states = self.ladder.init_ladder_states(hidden_states)
        for i, (layer_module,ladder_module) in enumerate(zip(self.layer,self.ladder.ladder_layers)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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
                ladder_outputs = ladder_module(
                    layer_outputs[1],
                    ladder_hidden_states,
                )

            hidden_states = layer_outputs[0]
            ladder_hidden_states = ladder_outputs
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        ladder_hidden_states = self.ladder.output_last_logits(ladder_hidden_states)
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
            last_hidden_state=ladder_hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LadderBertLayer(BertLayer):
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,intermediate_output
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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = layer_output + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

class LadderBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = LadderBertEncoder(config,)
        self.post_init()
        
                

class LadderBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = LadderBertModel(config)
        self.post_init()
    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        # freeze backbone model
        self.freeze()
    
    def freeze(self):
        for n,m in self.named_parameters():
            if 'ladder' not in n:
                m.requires_grad_(False)
