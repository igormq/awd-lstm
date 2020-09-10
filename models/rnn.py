import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self,
                 rnn_type,
                 num_tokens,
                 num_embedding,
                 num_hidden,
                 num_layers,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.num_tokens = num_tokens
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_embedding)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnns = torch.nn.ModuleList(
                getattr(nn, rnn_type)(
                    num_embedding if l == 0 else num_hidden,
                    num_hidden if l != num_layers - 1 else (
                        num_embedding if tie_weights else num_hidden),
                    num_layers=1,
                    dropout=dropout) for l in range(num_layers))
        else:
            try:
                nonlinearity = {
                    'RNN_TANH': 'tanh',
                    'RNN_RELU': 'relu'
                }[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnns = torch.nn.ModuleList(
                nn.RNN(num_embedding if l == 0 else num_hidden,
                       num_hidden if l != num_layers - 1 else (
                           num_embedding if tie_weights else num_hidden),
                       num_layers=1,
                       nonlinearity=nonlinearity,
                       dropout=dropout) for l in range(num_layers))
        self.decoder = nn.Linear(num_embedding if tie_weights else num_hidden, num_tokens, bias=False)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        self.num_embedding = num_embedding

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))

        output = emb
        for l, rnn in enumerate(self.rnns):
            output, hidden[l] = rnn(output, hidden[l])
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.num_tokens)
        return F.log_softmax(decoded, dim=-1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return [[weight.new_zeros(
                1, batch_size, self.num_hidden if l != self.num_layers - 1 else
                (self.num_embedding if self.tie_weights else self.num_hidden)),
                     weight.new_zeros(
                         1, batch_size,
                         self.num_hidden if l != self.num_layers - 1 else
                         (self.num_embedding
                          if self.tie_weights else self.num_hidden))]
                    for l in range(self.num_layers)]
        else:
            return [weight.new_zeros(
                    1, batch_size,
                    self.num_hidden if l != self.num_layers - 1 else
                    (self.num_embedding
                     if self.tie_weights else self.num_hidden))
                for l in range(self.num_layers)
            ]
