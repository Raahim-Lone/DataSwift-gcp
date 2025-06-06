# modules/plan2vec.py
#!/usr/bin/env python3
"""
Plan2VecEncoder — GINEConv w/ edge-type embeddings, graph summaries,
SQL-text bag-of-tokens, ready to append LLM+PCA features.
"""
import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool
from torch_scatter import scatter_add, scatter_mean
from modules.op_maps import EDGE_TYPE_MAP

class Plan2VecEncoder(nn.Module):
    def __init__(
        self,
        num_op_types: int,
        numeric_dim: int,
        vocab_size: int,
        text_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        out_dim: int = 512,
    ):
        super().__init__()
        self.out_dim = out_dim

        # — operator & edge embeddings
        self.op_embed   = nn.Embedding(num_op_types, 32)
        self.edge_embed = nn.Embedding(len(EDGE_TYPE_MAP), 16)

        # — build GINEConv layers (edge_dim=16)
        mlp1 = nn.Sequential(
            nn.Linear(32 + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([GINEConv(mlp1, train_eps=True, edge_dim=16)])
        for _ in range(num_layers - 1):
            mlpN = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlpN, train_eps=True, edge_dim=16))

        self.act = nn.LeakyReLU(0.1)

        # — SQL-text bag-of-tokens encoder
        self.token_embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)

        # — final MLP: [hidden_dim + 3 summaries + text_dim] → hidden_dim → out_dim
        summary_dim = 3
        mlp_in = hidden_dim + summary_dim + text_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data, sql_ids, sql_mask):
        # 1) initial node features
        op_ids   = data.x[:, 0].long()
        nums     = data.x[:, 1:]
        edge_emb = self.edge_embed(data.edge_attr)  # (E,16)
        h = torch.cat([self.op_embed(op_ids), nums], dim=1)

        # 2) GINEConv stack w/ conditional residual + LayerNorm
        for conv in self.convs:
            h_in = h
            h = conv(h, data.edge_index, edge_emb)
            h = nn.LayerNorm(h.size(-1)).to(h.device)(h)
            if h_in.size(-1) == h.size(-1):  # only when dims match
                h = self.act(h + h_in)
            else:
                h = self.act(h)

        # 3) global add pool
        g = global_add_pool(h, data.batch)  # (B, hidden_dim)

        # 4) graph‐level summaries: node count, avg fanout, avg cost
        batch       = data.batch
        ones        = torch.ones(batch.size(0), device=g.device)
        node_count  = scatter_add(ones, batch, dim=0)
        fanout_mean = scatter_mean(data.x[:,5], batch, dim=0)
        cost_mean   = scatter_mean(data.x[:,4], batch, dim=0)
        graph_sum   = torch.stack([node_count, fanout_mean, cost_mean], dim=1)

        # 5) SQL‐text embedding
        emb    = self.token_embed(sql_ids)       # (B, L, text_dim)
        mask   = sql_mask.unsqueeze(-1)          # (B, L, 1)
        summed = (emb * mask).sum(1)             # (B, text_dim)
        lengths= mask.sum(1).clamp(min=1)        # (B, 1)
        text_feat = summed / lengths             # (B, text_dim)

        # 6) concat & project
        cat = torch.cat([g, graph_sum, text_feat], dim=1)
        return self.mlp(cat)

