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


class MultiHopWordEmbedding(Module):
    def __init__(self, word2vec_model, vec2vec_models, routing=None):
        super().__init__()
        if not isinstance(vec2vec_models, list):
            vec2vec_models = [vec2vec_models]

        self.word2vec_model = word2vec_model
        self.vec2vec_models = ModuleList(vec2vec_models)
        self.routing = routing

    def forward(self, *args, **kwargs):
        # Get output from word2vec model
        outputs = [self.word2vec_model(*args, **kwargs)]
        for i in range(len(self.vec2vec_models)):
            # Get inputs for current vec2vec model
            if isinstance(self.routing[i], list):
                inputs = [outputs[idx] for idx in self.routing[i]]
            else:
                inputs = outputs[self.routing[i]]
            # Get output from current vec2vec model
            if isinstance(inputs, list):
                outputs.append(self.vec2vec_models[i](*[outputs[inp] for inp in inputs]))
            else:
                outputs.append(self.vec2vec_models[i](outputs[inputs]))
        return outputs


class HopThroughPhoneticSpaceWordEmbedding(MultiHopWordEmbedding):
    def __init__(self, word2vec_model, phonetic_model, semantic_model):
        super().__init__(word2vec_model, [phonetic_model, semantic_model])


class SummationPhoneticSpaceWordEmbedding(MultiHopWordEmbedding):
    def __init__(self, word2vec_model, phonetic_model, semantic_model1, semantic_model2):
        super().__init__(word2vec_model, [phonetic_model, semantic_model1, semantic_model2], routing=[0, 0, 1])

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs.append(outputs[-1] + outputs[-2])
        return outputs


class AveragingPhoneticSpaceWordEmbedding(MultiHopWordEmbedding):
    def __init__(self, word2vec_model, phonetic_model, semantic_model1, semantic_model2):
        super().__init__(word2vec_model, [phonetic_model, semantic_model1, semantic_model2], routing=[0, 0, 1])

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs.append((outputs[-1] + outputs[-2]) / 2)
