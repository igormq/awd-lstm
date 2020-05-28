import torch

from typing import Tuple


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class BPTTTensorDataset(torch.utils.data.IterableDataset):
    """ WARNING: it should be used with no automatic batching data loader, because the batching is already done here

    Args:
        data (torch.Tensor): 1-d tensor containing the samples
        batch_size (int): desired batch size
        bttp (int): the backpropagation through time number of samples. Default: 70
        random (bool): If `true` it will vary the size of bptt on each iteration. Default: `true`

    Returns:
        tuple (torch.Tensor, torch.Tensor) containing the source and target (which is the source shifted by 1) to use in language modelling
    """
    def __init__(self, data: torch.Tensor, batch_size: int, bptt: int = 70, random_bptt: bool = True):
        super().__init__()
        
        self.bptt = bptt

        num_batch = data.shape[0] // batch_size
        self.data = data.narrow(0, 0, num_batch * batch_size)
        self.data = self.data.view(batch_size, -1)
        self.random_bptt = random_bptt

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        i = 0
        while i < self.data.shape[1] - 1 - 1:

            if not self.random_bptt:
                seq_len = self.bptt
            else:
                bptt = self.bptt if torch.rand(1) < 0.95 else self.bptt / 2.
                # Prevent excessively small or negative sequence lengths
                seq_len = max(5, int(torch.distributions.Normal(bptt, 5).sample()))

            seq_len = min(seq_len, self.data.shape[1] - 1 - i)

            data = self.data[:, i:i + seq_len]
            target = self.data[:, i + 1:i + 1 + seq_len]

            yield data, target.contiguous().view(-1)
            i += seq_len
