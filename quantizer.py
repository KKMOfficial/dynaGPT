import torch
import torch.nn as nn
from einops import einsum


class Quantizer(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 beta=0.25,
                 ):
        super(Quantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.beta = beta
    
    def forward(self, x):
        B, C, D = x.shape
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape((-1, x.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_loss = codebook_loss + self.beta*commitment_loss
        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, C, D))
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2)))
        return quant_out, quantize_loss, min_encoding_indices


def get_quantizer(num_embeddings, embedding_dim):
    quantizer = Quantizer(
        num_embeddings,
        embedding_dim,
    )
    return quantizer

