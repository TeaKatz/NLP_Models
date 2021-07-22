import torch

from torch.nn import Module, ModuleList, Linear


class NonlinearTransformation(Module):
    def __init__(self, source_dim, intermediate_dim=None, destination_dim=None, intermediates_num=1, activation="leaky_relu", activation_kwargs=None):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = source_dim
        if destination_dim is None:
            destination_dim = source_dim
        
        self.activation = activation
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.source2inter = Linear(source_dim, intermediate_dim)
        self.inters = ModuleList([Linear(intermediate_dim, intermediate_dim) for _ in range(intermediates_num)])
        self.inter2dest = Linear(intermediate_dim, destination_dim)

    def activate(self, inputs):
        if self.activation == "relu":
            return torch.nn.functional.relu(inputs)
        elif self.activation == "leaky_relu":
            return torch.nn.functional.leaky_relu(inputs, **self.activation_kwargs)
        elif self.activation == "elu":
            return torch.nn.functional.elu(inputs, **self.activation_kwargs)
        elif self.activation == "selu":
            return torch.nn.functional.selu(inputs)
        elif self.activation == "celu":
            return torch.nn.functional.celu(inputs, **self.activation_kwargs)
        elif self.activation == "softplus":
            return torch.nn.functional.softplus(inputs, **self.activation_kwargs)

    def forward(self, inputs):
        """
        inputs: (*, source_dim)

        Return
        outputs: (*, destination_dim)
        """
        outputs = self.activate(self.source2inter(inputs))
        for i in range(len(self.inters)):
            outputs = self.activate(self.inters[i](outputs))
        outputs = self.inter2dest(outputs)
        return outputs


class PassThroughAuxiliarySpaceWordEmbedding(Module):
    def __init__(self, primary_embedding, primary2auxiliary, auxiliary2target):
        super().__init__()
        self.primary_embedding = primary_embedding
        self.primary2auxiliary = primary2auxiliary
        self.auxiliary2target = auxiliary2target

    def forward(self, *args, **kwargs):
        pri_vecs = self.primary_embedding(*args, **kwargs)
        aux_vecs = self.primary2auxiliary(pri_vecs)
        tar_vecs = self.auxiliary2target(aux_vecs)
        return tar_vecs


class SummationAuxiliarySpaceWordEmbedding(Module):
    def __init__(self, primary_embedding, primary2auxiliary, primary2target, auxiliary2target):
        super().__init__()
        self.primary_embedding = primary_embedding
        self.primary2auxiliary = primary2auxiliary
        self.primary2target = primary2target
        self.auxiliary2target = auxiliary2target

    def forward(self, *args, **kwargs):
        pri_vecs = self.primary_embedding(*args, **kwargs)
        aux_vecs = self.primary2auxiliary(pri_vecs)
        tar1_vecs = self.primary2target(pri_vecs)
        tar2_vecs = self.auxiliary2target(aux_vecs)
        tar_vecs = tar1_vecs + tar2_vecs
        return tar_vecs


class AveragingAuxiliarySpaceWordEmbedding(Module):
    def __init__(self, primary_embedding, primary2auxiliary, primary2target, auxiliary2target):
        super().__init__()
        self.primary_embedding = primary_embedding
        self.primary2auxiliary = primary2auxiliary
        self.primary2target = primary2target
        self.auxiliary2target = auxiliary2target

    def forward(self, *args, **kwargs):
        pri_vecs = self.primary_embedding(*args, **kwargs)
        aux_vecs = self.primary2auxiliary(pri_vecs)
        tar1_vecs = self.primary2target(pri_vecs)
        tar2_vecs = self.auxiliary2target(aux_vecs)
        tar_vecs = (tar1_vecs + tar2_vecs) / 2
        return tar_vecs
