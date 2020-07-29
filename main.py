"""
Example template for defining a system.
"""
from argparse import ArgumentParser

import torch

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from datasets import BRTD
from datasets.bptt import BPTTTensorDataset
from model import WDLSTM
from utils import repackage_hidden

from metrics import PPL, BPC


class AWDLSTM(LightningModule):
    def __init__(self,
                 hparams: dict(),
                 **kwargs) -> 'LightningTemplateModel':
        # init superclass
        super().__init__(**kwargs)

        self.hparams = hparams
        self.model = WDLSTM(self.hparams.num_tokens,
                            num_layers=self.hparams.num_layers,
                            num_hidden=self.hparams.num_hidden,
                            num_embedding=self.hparams.num_embedding,
                            tie_weights=self.hparams.tie_weights,
                            embedding_dropout=self.hparams.embedding_dropout,
                            input_dropout=self.hparams.input_dropout,
                            hidden_dropout=self.hparams.hidden_dropout,
                            output_dropout=self.hparams.output_dropout,
                            weight_dropout=self.hparams.weight_dropout)
        print(self.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.hiddens = None
        self.ppl = PPL()
        self.bpc = BPC()

    def forward(self, x, hiddens=None):
        self.seq_len = x.shape[1]
        return self.model(x, hiddens)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """ During training, we rescale the learning rate depending on the 
        length of the resulting sequence compared to the original specified 
        sequence length. The rescaling step is necessary as sampling arbitrary 
        sequence lengths with a fixed learning rate favors short sequences over 
        longer ones. This linear scaling rule has been noted as important for 
        training large scale minibatch SGD without loss of accu- racy (Goyal et 
        al., 2017) and is a component of unbiased truncated backpropagation 
        through time (Tallec & Ollivier, 2017).
        """
        temp_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = temp_lr * \
            self.seq_len / self.hparams.bptt
        super().backward(trainer, loss, optimizer, optimizer_idx)
        optimizer.param_groups[0]['lr'] = temp_lr

    def on_epoch_start(self):
        self.hiddens = None

    def training_step(self, batch, batch_idx):
        x, y = batch

        out, self.hiddens, (hs, dropped_hs) = self(x, self.hiddens)
        self.hiddens = repackage_hidden(self.hiddens)

        raw_loss = self.criterion(out, y)
        loss = raw_loss

        # The AR and TAR loss are only applied to the output of the final RNN layer, not to all layers

        # WARNING: It is implementing here \ell_2^2 instead of \ell_2
        # Activation Regularization
        if self.hparams.alpha:
            loss += self.hparams.alpha * dropped_hs[-1].pow(2).mean()

        # Temporal Activation Regularization (slowness)
        if self.hparams.beta:
            loss += self.hparams.beta * \
                (hs[-1][1:] - hs[-1][:-1]).pow(2).mean()

        logs = {'train_ppl': self.ppl(raw_loss), 'train_bpc': self.bpc(
            raw_loss), 'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def on_batch_end(self):
        self.hiddens = None

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out, self.hiddens, (hs, dropped_hs) = self(x, self.hiddens)
        self.hiddens = repackage_hidden(self.hiddens)

        loss = x.shape[1] * self.criterion(out, y)
        return {'loss': loss, 'seq_len': x.shape[1]}

    def validation_epoch_end(self, outputs):
        loss = 0
        total_seq_len = 0
        for o in outputs:
            loss += o['loss']
            total_seq_len += o['seq_len']

        loss = loss / total_seq_len

        logs = {
            'val_loss': loss,
            'val_ppl': self.ppl(loss),
            'val_bpc': self.bpc(loss)
        }

        self.hiddens = None
        return {'log': logs, 'progress_bar': logs}

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
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.learning_rate,
                                         weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         self.hparams.multi_step_lr_milestones,
                                                         gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--num-embedding', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--num-hidden', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--learning-rate', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--gradient-clip-val', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--output-dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--hidden-dropout', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--input-dropout', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--embedding-dropout', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--weight-dropout', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--weight-decay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--no-tie-weights', dest='tie_weights',  default=True, action='store_false',
                        help='if set, does not tie the input/output embedding weights')
    parser.add_argument('--multi-step-lr-milestones', nargs="+", type=int, default=[1e10],
                        help='When (which epochs) to divide the learning rate by 10')

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    datasets, vocab = BRTD.create(hparams.data)
    hparams.num_tokens = len(vocab)

    train_data = DataLoader(
        BPTTTensorDataset(datasets['train'], hparams.batch_size, hparams.bptt), batch_size=None, num_workers=1)

    val_data = DataLoader(
        BPTTTensorDataset(datasets['valid'], hparams.batch_size, hparams.bptt), batch_size=None, num_workers=1)

    model = AWDLSTM(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
