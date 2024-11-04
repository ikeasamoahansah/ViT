iimport torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=768, patch_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, X):
        assert X.shape[-1] % self.patch_size == 0, "Input dimensions must be divisible"
        out_conv = self.conv(X) # [1, 768, 14, 14]
        out_conv = self.flatten(out_conv) # [1, 768, 196]
        out_conv = out_conv.permute(0, 2, 1) # [1, 196, 768]
        return out_conv


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, X):
        X = self.layer(X)
        X, _ = self.attention(
            query=X,
            key=X,
            value=X,
            need_weights=False
        )
        return X


class MLPBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=mlp_size)
        self.linear2 = nn.Linear(in_features=mlp_size, out_features=embed_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.seq = nn.Sequential(
            self.linear1,
            self.gelu,
            self.dropout,
            self.linear2,
            self.dropout
        )

    def forward(self, X):
        X = self.norm(X)
        X = self.seq(X)
        return X


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, att_dropout=0, batch_first=True, mlp_size=3072, mlp_dropout=0.1, MSA: MultiheadSelfAttentionBlock=MultiheadSelfAttentionBlock, MLP: MLPBlock=MLPBlock):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attention = MSA(embed_dim, num_heads, att_dropout, batch_first)
        self.mlp = MLP(embed_dim, mlp_size, mlp_dropout)

    def forward(self, X):
        X = self.attention(X) + X
        X = self.mlp(X) + X
        return X


class ViT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, embed_dim=768, mlp_size=3072, num_heads=12, num_layers=12, patch_size=16, att_dropout=0, mlp_dropout=0.1, batch_first=True, embedding_dropout=0.1, num_classes=1000, patch_embed: PatchEmbedding=PatchEmbedding, msa: MultiheadSelfAttentionBlock=MultiheadSelfAttentionBlock, mlp: MLPBlock=MLPBlock, encoder:TransformerEncoderBlock=TransformerEncoderBlock):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.patch_size = patch_size
        assert self.embed_dim % self.patch_size == 0, "Image dimensions are wrong"
        self.num_layers = num_layers
        self.num_of_patches = (img_size * img_size) // (patch_size **2)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embed_dim),
                           requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1,
                            self.num_of_patches+1, embed_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = patch_embed(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embed_dim
        ) 
        self.vit = nn.Sequential(*[encoder(
            embed_dim, num_heads, att_dropout, 
            batch_first, mlp_size, mlp_dropout, msa, mlp)
        for _ in range(self.num_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, X):
        batch_size = X.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        X = self.patch_embedding(X)
        X = torch.cat((class_token, X), dim=1)
        X = self.position_embedding + X
        X = self.embedding_dropout(X)
        X = self.vit(X)
        X = self.classifier(X[:, 0])
        return X


















