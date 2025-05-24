import torch
import torch.nn as nn

class BBox3DDecoder(nn.Module):
    def __init__(
            self,
            num_queries=100,
            d_model=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        ):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoded_feats):
        memory = encoded_feats.unsqueeze(0)  # (1, N, D)
        queries = self.query_embed.weight.unsqueeze(0)  # (1, M, D)
        output = self.transformer_decoder(queries, memory)  # (1, M, D)
        return output.squeeze(0)  # (M, D)
