import torch


class ToNumber:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        return [self.vocab[t] for t in tokens]


class ToTensor:
    def __init__(self, dtype=torch.long):
        self.dtype = dtype

    def __call__(self, x):
        return torch.tensor(x).to(self.dtype)


class Sequential:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
