import torch
import torch.nn as nn

import torch.nn.functional as F

from collections import OrderedDict as odict


class ForwardWithDrop(object):
    def __init__(self, weights_names_ls, module, dropout_p,
                 original_module_forward):
        self.weights_names_ls = weights_names_ls
        self.module = module
        self.dropout_p = dropout_p
        self.original_module_forward = original_module_forward

    def __call__(self, *args,
                 **kwargs):  # the function formerly known as "forward_new"
        for name_param in self.weights_names_ls:
            param = self.module._parameters.get(name_param)
            param_with_droput = Parameter(torch.nn.functional.dropout(
                param, p=self.dropout_p, training=self.module.training),
                                          requires_grad=param.requires_grad)
            self.module._parameters.__setitem__(name_param, param_with_droput)

        return self.original_module_forward(*args, **kwargs)

def _weight_drop(module, weights_names_ls, dropout_p):

    original_module_forward = module.forward
    forward_with_drop = ForwardWithDrop(weights_names_ls, module, dropout_p, original_module_forward)
    setattr(module, 'forward', forward_with_drop)
    return module

# def _weight_drop(module, weights, dropout):
#     """
#     Helper for `WeightDrop`.
#     """

#     for name_w in weights:
#         w = getattr(module, name_w)
#         del module._parameters[name_w]
#         module.register_parameter(name_w + '_raw', torch.nn.Parameter(w))

#     original_module_forward = module.forward

#     def forward(*args, **kwargs):
#         for name_w in weights:
#             raw_w = getattr(module, name_w + '_raw')
#             w = torch.nn.functional.dropout(raw_w,
#                                             p=dropout,
#                                             training=module.training)
#             module._parameters[name_w] = w.retain_grad()

#         return original_module_forward(*args, **kwargs)

#     setattr(module, 'forward', forward)


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class LockedDropout(nn.Module):
    """ Variational Dropout (Gal & Ghahramani, 2016)

    Samples a binary dropout mask only once upon the first call and then to repeatedly use that locked dropout mask for all repeated connections within the forward and backward pass.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
            batch_first (bool): If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``True``

    """
    def __init__(self, p=0.5, batch_first=True):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):

        if not self.training or not self.p:
            return x

        x = x.clone()
        if self.batch_first:
            mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        else:
            mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        return 'p={}, batch_first={}'.format(self.p, self.batch_first)


class EmbeddedDropout(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, dropout=None, scale=None, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False,sparse=False, _weight=None):
        if padding_idx is None:
            padding_idx = -1

        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

        self.dropout = dropout
        self.scale = scale


    def forward(self, input):
        if self.dropout and not self.training:
            mask = self.weight.new() \
                              .resize_((self.weight.size(0), 1)) \
                              .bernoulli_(1 - self.dropout) \
                              .expand_as(self.weight) / (1 - self.dropout)
            masked_weight = mask * self.weight
        else:
            masked_weight = self.weight

        if self.scale:
            masked_weight = self.scale.expand_as(masked_weight) * masked_weight

        return F.embedding(
            input, masked_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = super().extra_repr()

        if self.dropout is not None:
            s += ', dropout={dropout}'
        if self.scale is not None:
            s += ', scale={scale}'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, dropout=None, scale=None, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False,sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            dropout (boolean, optional): Dropout to be applied to the embedding weights.
            scale (flaot, optional): scale factor to be applied to the masked weights
            padding_idx (int, optional): See module initialization documentation.
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            dropout=dropout,
            scale=scale,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding

class WDLSTM(nn.Module):
    """ Weight-Droppped LSTM
        It uses DropConnect on hidden-to-hidden weights

        Used jointly with ASGD the original authors [1] achieved a PP of 57.3 on Penn Treebank and 65.8 on WikiText-2 and in conjuntion with neural cache, they achieved a PP of 52.8 and 52, respectively.

        All experiments use a three-layer LSTM model with 1150 units in the hidden layer and an embedding of size 400. All embedings weights were uniformly initialized in the interval [-0.1, 0.1] and all other weights were initialized between [-1/sqrt(H), 1/sqrt(H)], where H is the hidden size.

        DropConnect was applied in all hidden-to-hidden weights (U^i, U^f, U^o, U^c)

        Variational dropout was applied in all other dropout operations,specifically using the same dropout mask for all inputs and outputs of the LSTM within a given forward and backward pass. Each example within the mini- batch uses a unique dropout mask, rather than a single dropout mask being used over all examples, ensuring di- versity in the elements dropped out.

        Embeding dropout was applied. This is equivalent to performing dropout on the embedding matrix at a word level, where the dropout is broadcast across all the word vector’s embedding. The remaining non-dropped-out word embeddings are scaled by 1/1-pe where pe is the probability of embedding dropout

        Weight tying (optional).

        Independent embedding size and hidden size.

        Activation Regularization (AR) and Temporal Activation Regularization (TAR). AR is defined as:
            alpha * L_2(m*h_t)

        where m is the dropout mask. TAR is defined as
            beta * L_2(h_t - h_{t+1})

        The values used for dropout on the word vectors, the output between LSTM layers, the output of the final LSTM layer, and embedding dropout where (0.4, 0.3, 0.4, 0.1) respectively. For the weight-dropped LSTM, a dropout of 0.5 was applied to the recurrent weight matrices. For WT2, we increase the input dropout to 0.65 to account for the increased vocabulary size.


        References:
            [1] - Merity, Stephen, Nitish Shirish Keskar, and Richard Socher. "Regularizing and optimizing LSTM language models.", 2017.
    """

    def __init__(self, num_tokens, num_layers=3, num_hidden=1150, num_embedding=400, tie_weights=True, embedding_dropout=0.1, input_dropout=0.4, hidden_dropout=0.3, output_dropout=0.4, weight_dropout=0.5, batch_first: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_embedding = num_embedding
        self.num_tokens = num_tokens
        self.tie_weights = tie_weights
        self.batch_first = batch_first

        self.encoder = EmbeddedDropout(num_tokens,
                                       num_embedding,
                                       dropout=embedding_dropout)
        self.input_locked_dropout = LockedDropout(input_dropout,
                                                  batch_first=batch_first)



        self.rnns = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict([
                    ['rnn', WeightDropLSTM(num_embedding if l == 0 else num_hidden,
                           num_hidden if l != num_layers - 1 else
                           (num_embedding if tie_weights else num_hidden),
                           num_layers=1,
                           weight_dropout=weight_dropout,
                           batch_first=batch_first)],
                    ['locked_dropout', LockedDropout(hidden_dropout,
                                                   batch_first=batch_first) if l != num_layers - 1 else None]
                ])
            for l in range(num_layers)]
        )

        self.output_locked_dropout = LockedDropout(output_dropout,
                                                   batch_first=batch_first)

        self.decoder = nn.Linear(
            num_embedding if tie_weights else num_hidden, num_tokens)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight


        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        torch.nn.init.zeros_(self.decoder.bias)
        torch.nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.shape[0 if self.batch_first else 1])

        embedding = self.encoder(x)
        embedding = self.input_locked_dropout(embedding)

        out = embedding
        # For Activation Regularization (AR)
        # m \odot ht
        m_hs = []

        # For Temporal Activation Regularization (TAR)
        # h_t - h_{t + 1}
        hs = []

        h = [[hn, cn] for (hn, cn) in h]
        for l, layer in enumerate(self.rnns):
            rnn, locked_dropout = layer['rnn'], layer['locked_dropout']
            out, h[l] = rnn(out, h[l])
            hs.append(out)

            if l != self.num_layers - 1:
                out = locked_dropout(out)
                m_hs.append(out)

        out = self.output_locked_dropout(out)
        m_hs.append(out)

        out = self.decoder(out)
        out = out.view(-1, self.num_tokens)

        return F.log_softmax(out, dim=1), h, (hs, m_hs)

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters())

        return [(weight.new_zeros(1, batch_size, self.num_hidden if l != self.num_layers - 1 else (self.num_embedding if self.tie_weights else self.num_hidden)), weight.new_zeros(1, batch_size, self.num_hidden if l != self.num_layers - 1 else (self.num_embedding if self.tie_weights else self.num_hidden))) for l in range(self.num_layers)]