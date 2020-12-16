import os
import re
from collections import Counter

from tqdm import tqdm
import torch

from torchtext.vocab import Vocab

import hashlib

from .functional import ToNumber, ToTensor, Sequential

_patterns = [
    r'\'', r'\"', r'\.', r'<br \/>', r',', r'\(', r'\)', r'\!', r'\?', r'\;',
    r'\:', r'\s+'
]

_replacements = [
    ' \'  ', '', ' . ', ' ', ' , ', ' ( ', ' ) ', ' ! ', ' ? ', ' ', ' ', ' '
]

_patterns_dict = list(
    (re.compile(p), r) for p, r in zip(_patterns, _replacements))


def _build_vocab(data, transforms, min_freq=3, max_size=200000):
    tok_list = map(lambda x: transforms(x), data)

    counter = Counter()
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in tok_list:
            counter.update(tokens)
            t.update(1)
    vocab = Vocab(counter,
                  max_size=max_size,
                  min_freq=min_freq,
                  specials=('<unk>', '<eos>', '<eos>', '<pad>'))
    return vocab


def _basic_pt_word_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for Brazilian Portuguese words as 
    follows:
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
    return ['<sos>'] + line.split() + ['<eos>']


def _basic_pt_char_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return list(line)


class BRTD:
    @staticmethod
    def create(root,
               vocab=None,
               tokenizer=_basic_pt_word_normalize,
               min_freq=3,
               max_size=200000):
        vocab = vocab
        transforms = None

        datasets = {}

        if isinstance(vocab, str):
            vocab_sha256 = hashlib.sha256(open(vocab,
                                               'rb').read()).hexdigest()[:6]
            vocab = torch.load(vocab)
            transforms = Sequential(ToNumber(vocab), ToTensor(torch.long))

        for subset in ['train', 'valid', 'test']:
            with open(os.path.join(root, f"{subset}.txt")) as f:
                data = [l.strip() for l in f.readlines()]

            data_sha256 = hashlib.sha256(''.join(data).encode()).hexdigest()

            if vocab is None and subset == 'train':
                tokenizer_sha256 = hashlib.sha256(
                    f'{tokenizer.__name__}-{min_freq}-{max_size}'.encode(
                    )).hexdigest()[:6]
                vocab_file = os.path.join(
                    root, f'{data_sha256}.{tokenizer_sha256}.vocab')
                if os.path.exists(vocab_file):
                    vocab = torch.load(vocab_file)
                else:
                    print('Building the vocabulary...')
                    vocab = _build_vocab(data,
                                         tokenizer,
                                         min_freq=min_freq,
                                         max_size=max_size)
                    torch.save(vocab, vocab_file)
                transforms = Sequential(ToNumber(vocab), ToTensor(torch.long))
                vocab_sha256 = hashlib.sha256(open(
                    vocab_file, 'rb').read()).hexdigest()[:6]

            tokens_file = os.path.join(
                root, f'{data_sha256}.{vocab_sha256}.{subset}.tokens')
            if os.path.exists(tokens_file):
                datasets[subset] = torch.load(tokens_file)
            else:
                print(f'Tokenizing {subset} data')
                data = [
                    t for line in map(lambda line: tokenizer(line), data)
                    for t in line
                ]
                datasets[subset] = transforms(data)
                torch.save(datasets[subset], tokens_file)

        return datasets, vocab
