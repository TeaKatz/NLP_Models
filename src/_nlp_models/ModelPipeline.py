import torch


class ModelPipeline:
    def __init__(self, preprocessor, model, return_numpy=True):
        self.preprocessor = preprocessor
        self.model = model
        self.return_numpy = return_numpy

    def __call__(self, *args, **kwargs):
        inputs = self.preprocessor(*args, **kwargs)
        if isinstance(inputs, dict):
            inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        else:
            inputs = torch.tensor(inputs)

        if isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)

        if self.return_numpy:
            outputs = outputs.detach().numpy()
        return outputs
