import torch

from torch.nn import Module, Embedding, RNN, GRU, LSTM


class CharacterLevelWordSparseEncoding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, padding_idx=0, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.mode = mode

    def forward(self, token_ids):
        """
        token_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, num_embeddings) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, num_embeddings) otherwise
        """
        # (batch_size, words_num, word_length, num_embeddings)
        word_vecs = torch.nn.functional.one_hot(token_ids, self.num_embeddings)
        word_vecs[:, :, :, self.padding_idx] = 0

        if self.mode == "sum":
            # (batch_size, words_num, num_embeddings)
            word_vecs = torch.sum(word_vecs, dim=2)
        elif self.mode == "mean":
            # (batch_size, words_num, 1)
            divider = torch.sum(token_ids != 0, dim=-1, keepdim=True)
            # (batch_size, words_num, num_embeddings)
            word_vecs = torch.sum(word_vecs, dim=2)
            word_vecs = word_vecs / divider
        elif self.mode == "max":
            # (batch_size, words_num, num_embeddings)
            word_vecs = torch.max(word_vecs, dim=2)[0]
        return word_vecs


class CharacterLevelWordEmbedding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.mode = mode

        self.embedding = Embedding(num_embeddings=num_embeddings, 
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, embedding_dim) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, embedding_dim) otherwise
        """
        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.embedding(token_ids)

        if self.mode == "sum":
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.sum(word_vecs, dim=2)
        elif self.mode == "mean":
            # (batch_size, words_num, 1)
            divider = torch.sum(token_ids != 0, dim=-1, keepdim=True)
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.sum(word_vecs, dim=2)
            word_vecs = word_vecs / divider
        elif self.mode == "max":
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.max(word_vecs, dim=2)[0]
        return word_vecs


class PositionalCharacterLevelWordSparseEncoding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, padding_idx=0, max_positional=10, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.max_positional = max_positional
        self.mode = mode

    def forward(self, token_ids, position_ids):
        """
        token_ids: (batch_size, words_num, word_length)
        position_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, num_embeddings + max_positional) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, num_embeddings + max_positional) otherwise
        """
        # (batch_size, words_num, word_length, num_embeddings)
        word_vecs = torch.nn.functional.one_hot(token_ids, self.num_embeddings)
        word_vecs[:, :, :, self.padding_idx] = 0
        # (batch_size, words_num, word_length, max_positional)
        pos_vecs = torch.nn.functional.one_hot(position_ids, self.max_positional)
        pos_vecs[:, :, :, self.padding_idx] = 0
        # (batch_size, words_num, word_length, num_embeddings + max_positional)
        word_vecs = torch.cat([word_vecs, pos_vecs], dim=-1)

        if self.mode == "sum":
            # (batch_size, words_num, num_embeddings + max_positional)
            word_vecs = torch.sum(word_vecs, dim=2)
        elif self.mode == "mean":
            # (batch_size, words_num, 1)
            divider = torch.sum(token_ids != 0, dim=-1, keepdim=True)
            # (batch_size, words_num, num_embeddings + max_positional)
            word_vecs = torch.sum(word_vecs, dim=2)
            word_vecs = word_vecs / divider
        else:
            # (batch_size, words_num, num_embeddings + max_positional)
            word_vecs = torch.max(word_vecs, dim=2)[0]
        return word_vecs


class PositionalCharacterLevelWordEmbedding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, max_positional=10, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.mode = mode

        self.word_embedding = Embedding(num_embeddings=num_embeddings, 
                                        embedding_dim=embedding_dim, 
                                        padding_idx=padding_idx)

        self.pos_embedding = Embedding(num_embeddings=max_positional,
                                       embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)

    def forward(self, token_ids, position_ids):
        """
        token_ids: (batch_size, words_num, word_length)
        position_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, embedding_dim) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, embedding_dim) otherwise
        """
        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.word_embedding(token_ids) + self.pos_embedding(position_ids)

        if self.mode == "sum":
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.sum(word_vecs, dim=2)
        elif self.mode == "mean":
            # (batch_size, words_num, 1)
            divider = torch.sum(token_ids != 0, dim=-1, keepdim=True)
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.sum(word_vecs, dim=2)
            word_vecs = word_vecs / divider
        else:
            # (batch_size, words_num, embedding_dim)
            word_vecs = torch.max(word_vecs, dim=2)[0]
        return word_vecs

    
class RNNCharacterLevelWordEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, num_layers=1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_layers = num_layers

        self.embedding = Embedding(num_embeddings=num_embeddings, 
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx)

        self.rnn = RNN(input_size=embedding_dim,
                       hidden_size=embedding_dim,
                       num_layers=num_layers,
                       batch_first=True)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, embedding_dim) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, embedding_dim) otherwise
        """
        batch_size, words_num, word_length = token_ids.shape

        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.embedding(token_ids)

        # (1, batch_size, embedding_dim)
        h0 = torch.zeros(1, batch_size * words_num, self.embedding_dim)
        
        # (batch_size * words_num, word_length, embedding_dim)
        word_vecs = word_vecs.view(batch_size * words_num, word_length, -1)
        word_vecs, _ = self.rnn(word_vecs, h0)

        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = word_vecs.view(batch_size, words_num, word_length, -1)
        return word_vecs[:, :, -1]


class GRUCharacterLevelWordEmbedding(RNNCharacterLevelWordEmbedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, num_layers=1):
        super().__init__(num_embeddings, embedding_dim, padding_idx, num_layers)
        self.rnn = GRU(input_size=embedding_dim,
                       hidden_size=embedding_dim,
                       num_layers=num_layers,
                       batch_first=True)


class LSTMCharacterLevelWordEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, num_layers=1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_layers = num_layers

        self.embedding = Embedding(num_embeddings=num_embeddings, 
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx)

        self.rnn = LSTM(input_size=embedding_dim,
                        hidden_size=embedding_dim,
                        num_layers=num_layers,
                        batch_first=True)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, embedding_dim) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, embedding_dim) otherwise
        """
        batch_size, words_num, word_length = token_ids.shape

        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.embedding(token_ids)

        # (1, batch_size, embedding_dim)
        h0 = torch.zeros(1, batch_size * words_num, self.embedding_dim)
        c0 = torch.zeros(1, batch_size * words_num, self.embedding_dim)
        
        # (batch_size * words_num, word_length, embedding_dim)
        word_vecs = word_vecs.view(batch_size * words_num, word_length, -1)
        word_vecs, _ = self.rnn(word_vecs, (h0, c0))

        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = word_vecs.view(batch_size, words_num, word_length, -1)
        return word_vecs[:, :, -1]
