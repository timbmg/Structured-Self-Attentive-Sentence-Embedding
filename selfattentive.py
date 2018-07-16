import torch
import torch.nn as nn

from attention import Attention
from encoder import Encoder

class SelfAttentive(nn.Module):

    def __init__(self, num_embeddings, num_embedding_hidden, num_encoder_hidden, num_classifier_hidden,
        num_attention_hidden, num_hops, rnn_cell='lstm', bidirectional=True):

        super().__init__()

        self.rnn_cell = rnn_cell
        self.num_classifier_hidden = num_classifier_hidden

        self.encoder = Encoder(num_embeddings, num_embedding_hidden, num_encoder_hidden, 
            rnn_cell=rnn_cell, bidirectional=bidirectional)

        self.attention = Attention(num_encoder_hidden*2, num_attention_hidden, num_hops)

        self.output = nn.Sequential(
            nn.Linear(num_hops * num_encoder_hidden*2, num_classifier_hidden),
            nn.ReLU(),
            nn.Linear(num_classifier_hidden, 5)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sequence, lengths, targets):

        encoder_hidden_states = self.encoder(sequence, lengths)
        self.attention_weights = self.attention(encoder_hidden_states, lengths)

        sentence_embedding = torch.bmm(self.attention_weights, encoder_hidden_states)
        self.sentence_embedding = sentence_embedding.view(-1, self.attention.num_hops * self.encoder.rnn.hidden_size*2)

        predictions = self.output(self.sentence_embedding)
        self.predictions = predictions.view(-1, 5)

        loss, penalization_term = self.loss_fn(self.predictions, targets, self.attention_weights)
        
        accuracy = self.accuracy(self.predictions, targets)

        return loss, penalization_term, accuracy

    def loss_fn(self, predictions, targets, attention_weights):
        
        B = predictions.size(0)

        loss = self.criterion(predictions, targets)

        AAT = torch.bmm(attention_weights, attention_weights.transpose(1,2))
        I = torch.eye(self.attention.num_hops).unsqueeze(0).repeat(B, 1, 1)
        penalization_term = torch.norm(AAT - I) / B

        return loss, penalization_term

    def accuracy(self, predictions, targets):

        return ((predictions.topk(1)[1].squeeze(1) == targets).sum()).item() / targets.size(0)

    def save(self, file_name, **kwargs):

        params = dict()
        params['num_embeddings'] = self.encoder.embedding.num_embeddings
        params['num_embedding_hidden'] = self.encoder.embedding.embedding_dim
        params['num_encoder_hidden'] = self.encoder.rnn.hidden_size
        params['num_classifier_hidden'] = self.num_classifier_hidden
        params['num_attention_hidden'] = self.attention.num_attention_hidden
        params['num_hops'] = self.attention.num_hops
        params['rnn_cell'] = self.encoder.rnn.mode.lower()
        params['bidirectional'] = self.encoder.rnn.bidirectional
        params['state_dict'] = self.state_dict()

        for k,v in kwargs.items():
            assert k not in params
            params[k] = v

        torch.save(params, open(file_name, 'wb'))

    @classmethod
    def load(cls, file_name):
        params = torch.load(file_name)
        model = cls(params['num_embeddings'], params['num_embedding_hidden'], params['num_encoder_hidden'], 
            params['num_classifier_hidden'],params['num_attention_hidden'], params['num_hops'], 
            params['rnn_cell'], params['bidirectional'])
        model.load_state_dict(params['state_dict'])
        return model
