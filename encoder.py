import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, num_embeddings, num_embedding_hidden, num_encoder_hidden, rnn_cell='lstm', bidirectional=True):

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, num_embedding_hidden)

        if rnn_cell == 'lstm':
            rnn = nn.LSTM
        elif rnn_cell == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()

        self.rnn = rnn(num_embedding_hidden, num_encoder_hidden, bidirectional=bidirectional, batch_first=True)

    def forward(self, sequence, lenghts):

        emb = self.embedding(sequence)
        
        sorted_lengths, sorted_idx = torch.sort(lenghts, descending=True)
        emb = emb[sorted_idx]
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, sorted_lengths, batch_first=True)
        
        packed_outptus, _ = self.rnn(packed_emb)
        padded_outptus, _ = nn.utils.rnn.pad_packed_sequence(packed_outptus, batch_first=True)

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outptus = padded_outptus[reversed_idx]

        return padded_outptus