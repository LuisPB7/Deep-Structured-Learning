import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from variables import *
from utils import create_emb_layer, NLIDataset, PadSequence
import tqdm
from collections import OrderedDict

import pytorch_lightning as pl

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class SentenceEncoder(nn.Module):
    def __init__(self):
        super(SentenceEncoder, self).__init__()
        self.embedding_layer = create_emb_layer(non_trainable=True)
        self.LSTM1 = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.LSTM2 = nn.LSTM(HIDDEN_SIZE*3, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT_P)
            
    def init_hidden(self, batch_size):
        directions = 2
        return Variable(torch.zeros((directions, batch_size, HIDDEN_SIZE))).cuda(),\
               Variable(torch.zeros((directions, batch_size, HIDDEN_SIZE))).cuda()
    
    def forward(self, sentence, lengths):
        # Init hidden state for both Bi-LSTMs
        hidden_1 = self.init_hidden(sentence.size()[0])
        hidden_2 = self.init_hidden(sentence.size()[0])

        sentence = self.embedding_layer(sentence)
        sent_shortcut = sentence
        sentence = torch.nn.utils.rnn.pack_padded_sequence(sentence, lengths, batch_first=True, enforce_sorted=False)        
        s_rep, _ = self.LSTM1(sentence, hidden_1)
        s_rep, _ = torch.nn.utils.rnn.pad_packed_sequence(s_rep, batch_first=True)
        s_rep = self.dropout(s_rep)
        s_rep = torch.cat([s_rep, sent_shortcut], 2)
        s_rep = torch.nn.utils.rnn.pack_padded_sequence(s_rep, lengths, batch_first=True, enforce_sorted=False)
        s_rep, _ = self.LSTM2(s_rep, hidden_2)
        s_rep, _ = torch.nn.utils.rnn.pad_packed_sequence(s_rep, batch_first=True)
        s_rep = self.dropout(s_rep)

        s_rep = nn.MaxPool2d((s_rep.size()[1],1))(s_rep).squeeze(dim=1)
        return s_rep

class NLIModel(pl.LightningModule):

    def __init__(self):
        super(NLIModel, self).__init__()
        self.dropout = nn.Dropout(DROPOUT_P)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sent_encoder = SentenceEncoder()
        self.lin1 = nn.Linear(HIDDEN_SIZE*8, HIDDEN_SIZE)
        self.lin2 = nn.Linear(HIDDEN_SIZE, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, premises, p_lens, hyps, hyp_lens):
        premise_rep = self.sent_encoder(premises, p_lens)
        hyp_rep = self.sent_encoder(hyps, hyp_lens)

        # Concatenate, multiply, subtract (1)
        conc = torch.cat([premise_rep, hyp_rep], 1)
        mul = premise_rep * hyp_rep
        dif = torch.abs(premise_rep - hyp_rep)
        final = torch.cat([conc, mul, dif], 1)
        final = self.dropout(final)
        # Linear layers and softmax
        final = self.lin1(final)
        final = self.relu(final)
        final = self.dropout(final)
        final = self.lin2(final)
        final = self.softmax(final)
        return final

    def training_step(self, batch, batch_nb):
        # REQUIRED
        outputs = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda())
        labels = batch[4]
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(NLIDataset("snli_1.0_train.jsonl"), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, collate_fn=PadSequence())
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(NLIDataset("snli_1.0_test.jsonl"), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, collate_fn=PadSequence())
    
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(NLIDataset("snli_1.0_dev.jsonl"), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, collate_fn=PadSequence())
    
    def validation_step(self, batch, batch_nb):
    
        # implement your own
        out = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda())
        y = batch[4]
        loss_val = F.cross_entropy(out, y)
    
        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    
        # all optional...
        # return whatever you need for the collation function validation_end
        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': torch.tensor(val_acc), # everything must be a tensor
        })
    
        # return an optional dict
        return output
    
    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
    
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'val_loss': val_loss_mean.item()}
        }
        return results
    
    def test_step(self, batch, batch_nb):
    
        # implement your own
        out = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda())
        y = batch[4]
        loss_test = F.cross_entropy(out, y)
    
        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    
        # all optional...
        # return whatever you need for the collation function test_end
        output = OrderedDict({
            'test_loss': loss_test,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
        })
    
        # return an optional dict
        return output
    
    def test_end(self, outputs):
        """
        Called at the end of test to aggregate outputs
        :param outputs: list of individual outputs of each test step
        :return:
        """
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']
    
        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
    
        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'test_loss': val_loss_mean.item()}
        }
        return results    
