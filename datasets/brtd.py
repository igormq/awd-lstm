import os
import re
from collections import Counter

from tqdm import tqdm
import torch

from torchtext.experimental.datasets import LanguageModelingDataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

from .bptt import BPTTTensorDataset


from .functional import ToNumber, ToTensor, Sequential


_patterns = [r'\'',
             r'\"',
             r'\.',
             r'<br \/>',
             r',',
             r'\(',
             r'\)',
             r'\!',
             r'\?',
             r'\;',
             r'\:',
             r'\s+']

_replacements = [' \'  ',
                 '',
                 ' . ',
                 ' ',
                 ' , ',
                 ' ( ',
                 ' ) ',
                 ' ! ',
                 ' ? ',
                 ' ',
                 ' ',
                 ' ']

_patterns_dict = list((re.compile(p), r)
                      for p, r in zip(_patterns, _replacements))


def _basic_pt_word_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for Brazilian Portuguese words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'text_transform
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def _build_vocab(data, transforms, min_freq=1, max_size=None):
    tok_list = map(lambda x: transforms(x), data)

    counter = Counter()
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in tok_list:
            counter.update(tokens)
            t.update(1)
    vocab = Vocab(counter, min_freq=3)
    return vocab


def _basic_pt_char_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return list(line)


def _setup_dataset(root_path, tokenizer=None, vocab=None,
                   data_select=('train', 'test', 'valid')):
    if tokenizer is None:
        tokenizer = _basic_pt_word_normalize

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'valid', 'test'))):
        raise TypeError(
            'Given data selection {} is not supported!'.format(data_select))

    with open(os.path.join(root_path, 'train.txt')) as f_train, open(os.path.join(root_path, 'valid.txt')) as f_valid, open(os.path.join(root_path, 'test.txt')) as f_test:
        train, test, valid = f_train.readlines(), f_test.readlines(), f_valid.readlines()
        raw_data = {'train': [txt for txt in train],
                    'valid': [txt for txt in valid],
                    'test': [txt for txt in test]}

    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")

        vocab = _build_vocab(raw_data['train'], tokenizer)

    transforms = Sequential(tokenizer, ToNumber(vocab), ToTensor(torch.long))
    data = {k: torch.cat(tuple(transforms(row) for row in data), axis=0)
            for k, data in raw_data.items()}
    return tuple(data[item] for item in data_select), vocab


def BRTD(root_path, level='word', **kwargs):

    if level == 'word':
        return _setup_dataset(root_path, **kwargs)
    elif level == 'char':
        kwargs.setdefault('tokenizer', _basic_pt_char_normalize)
        return _setup_dataset(root_path, **kwargs)
