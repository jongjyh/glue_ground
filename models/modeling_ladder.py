import torch.nn as nn
import torch
from transformers.activations import ACT2FN

class LadderLayers(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.r = config.r if hasattr(config,'r') else 8
        input_mode = config.input_mode if hasattr(config,'input_mode') else 'output'
        if input_mode == 'intermediate':
            self.in_dim = config.intermediate_size 
        elif input_mode == 'output':
            self.in_dim = config.hidden_size
        self.intermediate_dim = self.in_dim//self.r
        self.down = nn.Linear(self.in_dim,self.intermediate_dim)
        self.up = nn.Linear(self.intermediate_dim,config.hidden_size)
        self.intermediate = nn.Linear(self.intermediate_dim,self.intermediate_dim)
        self.alpha_gate = nn.Parameter(torch.zeros(1))
        self.b_tem = config.b_tem if hasattr(config,"b_tem") else 0.1
        self.u = config.u if hasattr(config,'u') else 0.2
        beta_mode = config.beta_mode if hasattr(config,"beta_mode") else 'parameter'
        if beta_mode == 'constant':
            self.register_buffer("beta_gate",torch.logit(torch.FloatTensor([ self.u ]))*self.b_tem)
        elif beta_mode == 'parameter':
            self.beta_gate = nn.Parameter(torch.logit(torch.FloatTensor([ self.u ]))*self.b_tem)
        self.temperature = config.a_tem if hasattr(config,"a_tem") else 0.1
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        self.add_layer_norm_before_adapter = False
        self.add_layer_norm_after_adapter = True
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(self.intermediate_dim, eps=config.layer_norm_eps)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(self.in_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self,ladder_hidden_states,backbone_hidden_states=None):
        # mixing the input by gate variable
        mu = torch.sigmoid(self.alpha_gate/self.temperature)
        if backbone_hidden_states is not None:
            backbone_hidden_states = self.dropout(backbone_hidden_states)
            down_hidden_states = self.down(backbone_hidden_states)
        else:
            down_hidden_states = ladder_hidden_states
        # if self.add_layer_norm_before_adapter:
        #     ladder_hidden_states = self.pre_layer_norm(ladder_hidden_states)
        inputs = mu * down_hidden_states + (1-mu) * ladder_hidden_states
        
        if self.add_layer_norm_before_adapter:
            inputs = self.pre_layer_norm(inputs)
        
        inputs = self.intermediate_act_fn(inputs)
        outputs_states= self.intermediate(inputs)
        # outputs_states = self.dropout(outputs_states)
        
            
        if backbone_hidden_states is not None:
            backbone_output_states = self.up(outputs_states)
            if self.add_layer_norm_after_adapter:
                backbone_output_states = self.post_layer_norm(backbone_output_states)
            # backbone_output_states = self.dropout(backbone_output_states)
        else:
            backbone_output_states = None
        return outputs_states,backbone_output_states
        
class Ladder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        input_mode = config.input_mode if hasattr(config,'input_mode') else 'output'
        self.num_layers = config.num_layers if hasattr(config,'num_layers') else config.num_hidden_layers
        self.ladder_layers = nn.ModuleList([LadderLayers(config) for _ in range(self.num_layers)])
        self.r = config.r if hasattr(config,'r') else 8
        if input_mode == 'intermediate':
            self.in_dim = config.intermediate_size
        elif input_mode == 'output':
            self.in_dim = config.hidden_size
        self.intermediate_dim = self.in_dim//self.r
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





class TFMLadderLayers(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.r = config.r if hasattr(config,'r') else 8
        input_mode = config.input_mode if hasattr(config,'input_mode') else 'output'
        if input_mode == 'intermediate':
            self.in_dim = config.intermediate_size 
        elif input_mode == 'output':
            self.in_dim = config.hidden_size
        self.intermediate_dim = self.in_dim//self.r
        self.down = nn.Linear(self.in_dim,self.intermediate_dim)
        self.up = nn.Linear(self.intermediate_dim,config.hidden_size)
        self.intermediate = nn.Linear(self.intermediate_dim,self.intermediate_dim)
        self.alpha_gate = nn.Parameter(torch.zeros(1))
        self.b_tem = config.b_tem if hasattr(config,"b_tem") else 0.1
        self.u = config.u if hasattr(config,'u') else 0.2
        beta_mode = config.beta_mode if hasattr(config,"beta_mode") else 'parameter'
        if beta_mode == 'constant':
            self.register_buffer("beta_gate",torch.logit(torch.FloatTensor([ self.u ]))*self.b_tem)
        elif beta_mode == 'parameter':
            self.beta_gate = nn.Parameter(torch.logit(torch.FloatTensor([ self.u ]))*self.b_tem)
        self.temperature = config.a_tem if hasattr(config,"a_tem") else 0.1
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        self.add_layer_norm_before_adapter = False
        self.add_layer_norm_after_adapter = True
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(self.intermediate_dim, eps=config.layer_norm_eps)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(self.in_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self,ladder_hidden_states,backbone_hidden_states=None):
        # mixing the input by gate variable
        mu = torch.sigmoid(self.alpha_gate/self.temperature)
        if backbone_hidden_states is not None:
            backbone_hidden_states = self.dropout(backbone_hidden_states)
            down_hidden_states = self.down(backbone_hidden_states)
        else:
            down_hidden_states = ladder_hidden_states
        # if self.add_layer_norm_before_adapter:
        #     ladder_hidden_states = self.pre_layer_norm(ladder_hidden_states)
        inputs = mu * down_hidden_states + (1-mu) * ladder_hidden_states
        
        if self.add_layer_norm_before_adapter:
            inputs = self.pre_layer_norm(inputs)
        
        inputs = self.intermediate_act_fn(inputs)
        outputs_states= self.intermediate(inputs)
        # outputs_states = self.dropout(outputs_states)
        
            
        if backbone_hidden_states is not None:
            backbone_output_states = self.up(outputs_states)
            if self.add_layer_norm_after_adapter:
                backbone_output_states = self.post_layer_norm(backbone_output_states)
            # backbone_output_states = self.dropout(backbone_output_states)
        else:
            backbone_output_states = None
        return outputs_states,backbone_output_states
        
class TFMLadder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        input_mode = config.input_mode if hasattr(config,'input_mode') else 'output'
        self.num_layers = config.num_layers if hasattr(config,'num_layers') else config.num_hidden_layers
        self.ladder_layers = nn.ModuleList([TFMLadderLayers(config) for _ in range(self.num_layers)])
        self.r = config.r if hasattr(config,'r') else 8
        if input_mode == 'intermediate':
            self.in_dim = config.intermediate_size
        elif input_mode == 'output':
            self.in_dim = config.hidden_size
        self.intermediate_dim = self.in_dim//self.r
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



