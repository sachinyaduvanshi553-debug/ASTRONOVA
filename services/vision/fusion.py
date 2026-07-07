import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """
    Fuses spatial image features with temporal and physics embeddings
    using Cross Attention.
    """
    def __init__(self, spatial_dim=256, temporal_dim=256, physics_dim=128):
        super(FusionNetwork, self).__init__()
        
        # Combine temporal and physics into a single context vector
        self.context_dim = temporal_dim + physics_dim
        
        # Cross Attention: Query comes from spatial features, Key/Value from context
        # We'll use a multi-head attention mechanism
        self.num_heads = 4
        self.q_proj = nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1)
        self.k_proj = nn.Linear(self.context_dim, spatial_dim)
        self.v_proj = nn.Linear(self.context_dim, spatial_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=self.num_heads, batch_first=True)
        
        self.out_proj = nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, spatial_dim)

    def forward(self, spatial_features, temporal_embedding, physics_embedding):
        # spatial_features: (B, C, H, W)
        # temporal_embedding: (B, temporal_dim)
        # physics_embedding: (B, physics_dim)
        
        B, C, H, W = spatial_features.shape
        
        # Context vector
        context = torch.cat([temporal_embedding, physics_embedding], dim=1) # (B, context_dim)
        # Add sequence dimension for attention (sequence length 1)
        context = context.unsqueeze(1) # (B, 1, context_dim)
        
        # Project K, V
        K = self.k_proj(context) # (B, 1, C)
        V = self.v_proj(context) # (B, 1, C)
        
        # Project Q and reshape
        Q = self.q_proj(spatial_features) # (B, C, H, W)
        Q = Q.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)
        
        # Apply Cross Attention
        # query: (B, H*W, C), key: (B, 1, C), value: (B, 1, C)
        attn_out, _ = self.attn(Q, K, V) # (B, H*W, C)
        
        # Reshape back to spatial dimensions
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Residual connection and normalization
        out = self.norm(spatial_features + self.out_proj(attn_out))
        return out
