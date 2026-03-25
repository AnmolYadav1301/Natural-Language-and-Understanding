import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_embed = nn.Embedding(vocab_size, embed_dim)

        # to make rest compatible with both skip and bow
        self.embeddings = self.input_embed

    def forward(self, center, context, negative):
        center_embed = self.input_embed(center)
        context_embed = self.output_embed(context)
        neg_embed = self.output_embed(negative)

        pos_score = torch.sum(center_embed * context_embed, dim=1)
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()

        loss = - (F.logsigmoid(pos_score) + torch.sum(F.logsigmoid(-neg_score), dim=1))

        return loss.mean()   # ✔ MUST return scalar
        # return pos_score, neg_score