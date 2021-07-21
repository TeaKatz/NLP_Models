import torch

from torch.nn import Module, Embedding


class CharacterLevelWordEmbedding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.mode = mode

        self.l_embedding = Embedding(num_embeddings=num_embeddings, 
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
        word_vecs = self.l_embedding(token_ids)

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


class PositionalCharacterLevelWordEmbedding(Module):
    mode_options = ["sum", "mean", "max", "none"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, max_positional=10, mode="sum"):
        assert mode in self.mode_options

        super().__init__()
        self.mode = mode

        self.l_word_embedding = Embedding(num_embeddings=num_embeddings, 
                                          embedding_dim=embedding_dim, 
                                          padding_idx=padding_idx)

        self.l_pos_embedding = Embedding(num_embeddings=max_positional,
                                         embedding_dim=embedding_dim,
                                         padding_idx=None)

    def forward(self, token_ids, position_ids):
        """
        token_ids: (batch_size, words_num, word_length)
        position_ids: (batch_size, words_num, word_length)

        Return
        word_vecs:  (batch_size, words_num, embedding_dim) if mode is ("sum", "mean", "max")
                    (batch_size, words_num, word_length, embedding_dim) otherwise
        """
        # (batch_size, words_num, word_length, embedding_dim)
        word_vecs = self.l_word_embedding(token_ids) + self.l_pos_embedding(position_ids)

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
