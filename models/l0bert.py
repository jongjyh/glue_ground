from typing import Optional,Union,Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import regex as re
from transformers.models.bert.modeling_bert import BertLayer,BertOutput,BertAttention,BertIntermediate,BertEncoder,BertModel,BertForSequenceClassification
from models import l0_layers
from models.l0_layers import L0Dense
class L0BertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = L0Dense(
                             in_features=config.intermediate_size,
                             out_features=config.hidden_size,
                             droprate_init=config.droprate_init,
                             lamba=config.lamb,
                             temperature=config.temperature
                             )
        self.dropout = torch.nn.Dropout(0.)
class L0BertIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.dense = L0Dense(
                             in_features=config.hidden_size,
                             out_features=config.intermediate_size,
                             droprate_init=config.droprate_init,
                             lamba=config.lamb,
                             temperature=config.temperature
                             )
        # self.dropout = torch.nn.Dropout(0.)
class L0BertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        if 'output' in config.prune_module:
            self.output = L0BertOutput(config)
        if 'intermediate' in config.prune_module:
            self.intermediate = L0BertIntermediate(config)
    
class L0BertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        parse = re.search("(\d+)-(\d+)",config.prune_module)
        start,end = int(parse[1]),int(parse[2])
        for i in range(start,end+1):
            self.layer[i] =L0BertLayer(config) 
        # self.layer = torch.nn.ModuleList([L0BertLayer(config) for _ in range(config.num_hidden_layers)])

class L0BertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = L0BertEncoder(config)
class L0BertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,N=32):
        super().__init__(config)
        self.bert = L0BertModel(config)
        self.post_init()
        self.arch_pmt = self.get_arch_pmt()
        self.N=N
    def get_arch_pmt(self):
        arch_pmt = []
        for n,m in self.named_modules():
            if 'output' in self.config.prune_module:
                if re.fullmatch(".*\.[0-9]+\.(?:output)",n) and isinstance(m,L0BertOutput):
                    arch_pmt.append(m.dense)
            if 'intermediate' in self.config.prune_module:
                if re.fullmatch(".*\.[0-9]+\.(?:intermediate)",n) and isinstance(m,L0BertIntermediate):
                    arch_pmt.append(m.dense)
        return arch_pmt
    
    def _init_weights(self, module):
        if isinstance(module,l0_layers.L0Dense):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        super()._init_weights(module)
    
    def use_init_weights(self):
        if self.arch_pmt is not None:
            for m in self.arch_pmt:
                self._init_weights(m)
                #kaiming init
                # m.reset_parameters()
    
    def get_flops(self):
        
        flops = []
        for m in self.arch_pmt:
            flops.append((m.count_expected_flops_and_l0()))
        return flops
                
    def regularization(self):
        regularization = 0.
        for layer in self.arch_pmt:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization
    
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        reg_term = self.regularization()
        output.loss = output.loss + (-reg_term.detach() + reg_term)
        # output.loss = output.loss 
        return output 