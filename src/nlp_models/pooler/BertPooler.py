import torch
from torch.nn import Module, Linear, Tanh, ReLU


class BertPooler:
    pooling_available = ["cls", "mean", "max"]

    def __init__(self, pooling_mode="cls"):
        assert pooling_mode in self.pooling_available
        self.pooling_mode = pooling_mode

    def __call__(self, hidden_state, attention_mask=None):
        """
        hidden_state: (batch_size, sequence_length, embedding_size)
        attention_mask: (batch_size, sequence_length)
        """
        if self.pooling_mode == "cls":
            outputs = hidden_state[:, 0]
        elif self.pooling_mode == "mean":
            assert attention_mask is not None, "Must provide attention_mask"
            outputs = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1)
            outputs = outputs / torch.sum(attention_mask, dim=1, keepdim=True)
        elif self.pooling_mode == "max":
            assert attention_mask is not None, "Must provide attention_mask"
            hidden_state[attention_mask.unsqueeze(-1) == 0] = -1e9
            outputs = torch.max(hidden_state, dim=1)
        return outputs


class BertPoolerWithProjection(Module):
    pooling_available = ["cls", "mean", "max"]
    activation_available = ["tanh", "relu"]

    def __init__(self, pooling_mode="cls", embedding_size=768, activation="tanh"):
        super().__init__()
        assert pooling_mode in self.pooling_available
        if activation is not None:
            activation in self.activation_available

        self.pooling_mode = pooling_mode
        self.linear = Linear(embedding_size, embedding_size)
        if activation == "tanh":
            self.activation = Tanh()
        elif activation == "relu":
            self.activation = ReLU()
        else:
            self.activation = None

    def forward(self, hidden_state, attention_mask=None):
        """
        hidden_state: (batch_size, sequence_length, embedding_size)
        attention_mask: (batch_size, sequence_length)
        """
        if self.pooling_mode == "cls":
            outputs = hidden_state[:, 0]
        elif self.pooling_mode == "mean":
            assert attention_mask is not None, "Must provide attention_mask"
            outputs = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1)
            outputs = outputs / torch.sum(attention_mask, dim=1, keepdim=True)
        elif self.pooling_mode == "max":
            assert attention_mask is not None, "Must provide attention_mask"
            hidden_state[attention_mask.unsqueeze(-1) == 0] = -1e9
            outputs = torch.max(hidden_state, dim=1)

        outputs = self.linear(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs