import torch


class IntegratedCLIP(torch.nn.Module):
    def __init__(self, cls, config, add_text_projection: bool = False):
        super().__init__()
        self.transformer = cls(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))

        if add_text_projection:
            embed_dim = config.hidden_size
            self.transformer.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        # Our Codex CLIP ignores attention_mask/position_ids (it computes positions internally).
        # Keep signature parity with HF so higher-level callers can route via the wrapper.
        outputs = self.transformer(input_ids, output_hidden_states=output_hidden_states)
        # outputs is an object with attributes: last_hidden_state, hidden_states, pooler_output
        if return_dict:
            return outputs
        # For tuple compatibility, return (last_hidden_state, pooler_output, hidden_states)
        return (outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states)
