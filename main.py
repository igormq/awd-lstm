"""
Example template for defining a system.
"""
import logging
from argparse import ArgumentParser
import math

import nni
import pytorch_lightning as pl
import torch
from nni.utils import merge_parameter
from loggers import TrainsLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from datasets import BRTD
from datasets.bptt import (BPTTBatchSampler, LanguageModelingDataset,
                           _collate_fn)
from metrics import BPC, PPL
from models import WDLSTM, RNNModel, TransformerModel
from utils import repackage_hidden

logger = logging.getLogger('lm')


class NNICallback(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module):
        if trainer.global_rank != 0:
            return

        if trainer.running_sanity_check:
            return

        if trainer.logged_metrics and 'val_loss' in trainer.logged_metrics:
            nni.report_intermediate_result(trainer.logged_metrics['val_loss'])

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        if trainer.running_sanity_check:
            return

        if trainer.logged_metrics and 'val_loss' in trainer.logged_metrics:
            nni.report_final_result(trainer.logged_metrics['val_loss'])


class AWDLSTM(LightningModule):
    def __init__(self, hparams: dict(), **kwargs) -> 'LightningTemplateModel':
        # init superclass
        super().__init__(**kwargs)

        self.hparams = hparams
        if self.hparams.model == 'awd':
            self.model = WDLSTM(
                self.hparams.num_tokens,
                num_layers=self.hparams.num_layers,
                num_hidden=self.hparams.num_hidden,
                num_embedding=self.hparams.num_embedding,
                tie_weights=self.hparams.tie_weights,
                embedding_dropout=self.hparams.embedding_dropout,
                input_dropout=self.hparams.input_dropout,
                hidden_dropout=self.hparams.hidden_dropout,
                output_dropout=self.hparams.output_dropout,
                weight_dropout=self.hparams.weight_dropout)
        elif self.hparams.model == 'rnn':
            self.model = RNNModel(self.hparams.rnn_type,
                                  self.hparams.num_tokens,
                                  num_embedding=self.hparams.num_embedding,
                                  num_hidden=self.hparams.num_hidden,
                                  num_layers=self.hparams.num_layers,
                                  dropout=self.hparams.dropout,
                                  tie_weights=self.hparams.tie_weights)
        elif self.hparams.model == 'transformer':
            self.model = TransformerModel(
                self.hparams.num_tokens,
                num_embedding=self.hparams.num_embedding,
                num_hidden=self.hparams.num_hidden,
                num_layers=self.hparams.num_layers,
                dropout=self.hparams.dropout,
                num_heads=self.hparams.num_heads)
        else:
            raise ValueError(f'Model {self.hparams.model} not recognized.')

        self.hiddens = None
        self.criterion = torch.nn.NLLLoss()
        self.ppl = PPL()
        self.bpc = BPC()
        self.avg_loss = 0

    def forward(self, x, hiddens=None):
        if self.hparams.model != 'transformer':
            return self.model(x, hiddens)
        return self.model(x)

    def on_train_epoch_start(self):
        self.train_len = len(
            self.train_dataloader().batch_sampler) * self.hparams.bptt
        if self.hparams.model != 'transformer':
            self.hiddens = self.model.init_hidden(self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.model == 'awd':
            self.hiddens = repackage_hidden(self.hiddens)
            out, self.hiddens, (hs, dropped_hs) = self(x, self.hiddens)
        elif self.hparams.model == 'rnn':
            self.hiddens = repackage_hidden(
                self.hiddens) if self.hiddens else self.hiddens
            out, self.hiddens = self(x, self.hiddens)
        elif self.hparams.model == 'transformer':
            out = self(x)

        raw_loss = self.criterion(out, y)
        loss = raw_loss

        # The AR and TAR loss are only applied to the output of the final
        # RNN layer, not to all layers

        if self.hparams.model == 'awd':
            # WARNING: It is implementing here \ell_2^2 instead of \ell_2
            # Activation Regularization
            if self.hparams.alpha > 0:
                loss += self.hparams.alpha * dropped_hs[-1].pow(2).mean()

            # Temporal Activation Regularization (slowness)
            if self.hparams.beta > 0:
                loss += self.hparams.beta * \
                    (hs[-1][1:] - hs[-1][:-1]).pow(2).mean()

        raw_loss = raw_loss.item()
        ppl = self.ppl(raw_loss)
        bpc = self.bpc(raw_loss)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        result.log('train_ppl', ppl, prog_bar=True)
        result.log('train_bpc', bpc, prog_bar=True)

        return result

    def on_validation_epoch_start(self):
        self.val_len = len(
            self.val_dataloader().batch_sampler) * self.hparams.bptt
        if self.hparams.model != 'transformer':
            self.hiddens = self.model.init_hidden(self.hparams.batch_size)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.model == 'awd':
            self.hiddens = repackage_hidden(self.hiddens)
            out, self.hiddens, (hs, dropped_hs) = self(x, self.hiddens)
        elif self.hparams.model == 'rnn':
            self.hiddens = repackage_hidden(
                self.hiddens) if self.hiddens else self.hiddens
            out, self.hiddens = self(x, self.hiddens)
        elif self.hparams.model == 'transformer':
            out = self(x)

        loss = self.criterion(out, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x: torch.sum(x) / self.val_len)
        result.log('val_bpc',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x:
                   (torch.sum(x) / self.val_len) / math.log(2))
        result.log('val_ppl',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x: torch.exp(torch.sum(x) / self.val_len))
        return result

    def on_test_epoch_start(self):
        self.test_len = len(
            self.test_dataloader().batch_sampler) * self.hparams.bptt
        if self.hparams.model != 'transformer':
            self.hiddens = self.model.init_hidden(self.hparams.batch_size)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.model == 'awd':
            self.hiddens = repackage_hidden(self.hiddens)
            out, self.hiddens, (hs, dropped_hs) = self(x, self.hiddens)
        elif self.hparams.model == 'rnn':
            self.hiddens = repackage_hidden(
                self.hiddens) if self.hiddens else self.hiddens
            out, self.hiddens = self(x, self.hiddens)
        elif self.hparams.model == 'transformer':
            out = self(x)

        loss = self.criterion(out, y)

        result = pl.EvalResult()
        result.log('test_loss',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x: torch.sum(x) / self.test_len)
        result.log('test_bpc',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x:
                   (torch.sum(x) / self.test_len) / math.log(2))
        result.log('test_ppl',
                   len(x) * loss,
                   prog_bar=True,
                   reduce_fx=lambda x: torch.exp(torch.sum(x) / self.test_len))
        return result

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.


        WARNING: The paper use a variation of ASGD, called non-monotonically
        triggered ASGD (Algorithm 1), which is not implemented yet, They used L
        to be the number of iterations in an epoch (i.e., after training epoch
        ends) and n=5.
        """
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, self.hparams.multi_step_lr_milestones, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num-embedding',
                            type=int,
                            default=400,
                            help='size of word embeddings')
        parser.add_argument('--num-hidden',
                            type=int,
                            default=1150,
                            help='number of hidden units per layer')
        parser.add_argument('--num-layers',
                            type=int,
                            default=3,
                            help='number of layers')
        parser.add_argument('--learning_rate',
                            '--learning-rate',
                            type=float,
                            default=30.0,
                            help='initial learning rate')
        parser.add_argument('--batch-size',
                            type=int,
                            default=80,
                            metavar='N',
                            help='batch size')
        parser.add_argument('--bptt',
                            type=int,
                            default=70,
                            help='sequence length')
        parser.add_argument('--output-dropout',
                            type=float,
                            default=0.4,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--hidden-dropout',
                            type=float,
                            default=0.3,
                            help='dropout for rnn layers (0 = no dropout)')
        parser.add_argument(
            '--input-dropout',
            type=float,
            default=0.65,
            help='dropout for input embedding layers (0 = no dropout)')
        parser.add_argument(
            '--embedding-dropout',
            type=float,
            default=0.1,
            help='dropout to remove words from embedding layer '
            '(0 = no dropout)')
        parser.add_argument(
            '--weight-dropout',
            type=float,
            default=0.5,
            help='amount of weight dropout to apply to the RNN hidden to '
            'hidden matrix')
        parser.add_argument(
            '--alpha',
            type=float,
            default=0,
            help='alpha L2 regularization on RNN activation (alpha = 0 means'
            ' no regularization)')
        parser.add_argument(
            '--beta',
            type=float,
            default=0,
            help='beta slowness regularization applied on RNN activiation '
            '(beta = 0 means no regularization)')
        parser.add_argument('--weight-decay',
                            type=float,
                            default=1.2e-6,
                            help='weight decay applied to all weights')
        parser.add_argument('--optimizer',
                            type=str,
                            default='sgd',
                            help='optimizer to use (sgd, adam)')
        parser.add_argument(
            '--no-tie-weights',
            dest='tie_weights',
            default=True,
            action='store_false',
            help='if set, does not tie the input/output embedding weights')
        parser.add_argument('--rnn-type',
                            choices=['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'],
                            default='LSTM')
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument(
            '--num-heads',
            type=int,
            default=2,
            help='the number of heads in the encoder/decoder of the '
            ' transformer model')
        return parser


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument('data',
                            type=str,
                            default='data/brtd/',
                            help='location of the data corpus')
        parser.add_argument('--vocab', default=None)
        parser.add_argument('--project-name',
                            type=str,
                            default='language-model')
        parser.add_argument('--task-name', default=None, type=str)
        parser.add_argument('--model',
                            type=str,
                            default='awd',
                            choices=['rnn', 'awd', 'transformer'])

        # add model specific args
        parser = AWDLSTM.add_model_specific_args(parser)

        # add args from trainer
        parser = Trainer.add_argparse_args(parser)

        hparams = parser.parse_args()

        task_name = hparams.task_name
        # most basic trainer, uses good defaults

        trains_logger = TrainsLogger(project_name=hparams.project_name,
                                     task_name=task_name,
                                     auto_connect_arg_parser={
                                         'rank': False,
                                         'tpu_cores': False
                                     })

        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)

        hparams = merge_parameter(hparams, tuner_params)
        logger.info(hparams)

        datasets, vocab = BRTD.create(hparams.data, vocab=hparams.vocab)
        hparams.num_tokens = len(vocab)

        train_dataset = LanguageModelingDataset(datasets['train'])
        train_batch_sampler = BPTTBatchSampler(train_dataset, hparams.bptt,
                                               hparams.batch_size)
        train_data = DataLoader(train_dataset,
                                num_workers=8,
                                batch_sampler=train_batch_sampler,
                                collate_fn=_collate_fn)

        valid_dataset = LanguageModelingDataset(datasets['valid'])
        valid_batch_sampler = BPTTBatchSampler(valid_dataset, hparams.bptt,
                                               hparams.batch_size)
        valid_data = DataLoader(valid_dataset,
                                num_workers=8,
                                batch_sampler=valid_batch_sampler,
                                collate_fn=_collate_fn)

        test_dataset = LanguageModelingDataset(datasets['test'])
        test_batch_sampler = BPTTBatchSampler(test_dataset, hparams.bptt,
                                              hparams.batch_size)
        test_data = DataLoader(test_dataset,
                               num_workers=8,
                               batch_sampler=test_batch_sampler,
                               collate_fn=_collate_fn)

        model = AWDLSTM(hparams)

        hparams.logger = trains_logger
        hparams.callbacks = [NNICallback()]
        trainer = Trainer.from_argparse_args(hparams)
        trainer.fit(model,
                    train_dataloader=train_data,
                    val_dataloaders=valid_data)
        trainer.test(model, test_dataloaders=test_data)

    except Exception as exception:
        logger.exception(exception)
        raise
