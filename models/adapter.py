from git import Object
import torch
import torch.nn as nn
import re
from transformers.activations import ACT2FN
from models.search_cells import NAS201SearchCell as SearchCell

from models.operation import OPS
# From https://github.com/rabeehk/compacter

ACT2FN["identity"] = lambda x: x
op_names=(
    "none",
    "avg_pool_3",
    "max_pool_3",
    "nor_conv_3",
    "nor_conv_5",
    "nor_conv_7",
    "dil_conv_3",
    "dil_conv_5",
    "dil_conv_7",
    "skip_connect"
)

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
        self.residual = config.normal_adapter_residual
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
        # self.init_weights()

        self.hardwts = torch.zeros_like(self.arch_parameters)
        self.index = torch.randint(0,self.hardwts.shape[-1],(self.hardwts.shape[0],1))
        self.hardwts.scatter_(-1,self.index,1.0)
    def init_weights(self):
        """Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function"""
        # self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.down_proj.bias.data.zero_()
        # self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.up_proj.bias.data.zero_()

    def forward(self, x):
        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbels) / self.tem
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
            return hardwts, index
        if False:
            # random select arch
            hardwts ,index= self.hardwts  , self.index
        else:
            hardwts, index = get_gumbel_prob(self.arch_parameters)
            self.tem -= self.tem_proportion.item() 
        if self.residual:
            output = self.cells.forward_gdas(x,hardwts,index) + x
        return output


class NASBertFF(nn.Module):
    def __init__(self, bertFF, config, transformer_config,):
        super().__init__()
        self.dense = bertFF.dense
        self.LayerNorm = bertFF.LayerNorm
        self.dropout = bertFF.dropout
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states) 
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def modify_with_adapters(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*layer.(?:11)", m_name):
            module.output = NASBertFF(
                                module.output,
                                config,
                                transformer.config,
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
    
