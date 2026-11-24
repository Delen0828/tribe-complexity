"""A compact Perceiver-style regressor for native fsaverage5 TRIBE responses."""
import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

PATCHES_PER_HEMISPHERE = 162
WIDTH = 64
LATENTS = 16
HEADS = 4
DEPTH = 2
DROPOUT = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.02
EPOCHS = (5, 10, 20, 32, 64, 96)


def cortical_patches():
    """Tile each sphere around its coarse mesh vertices without averaging values."""
    sphere = fetch_surf_fsaverage("fsaverage5")
    members, positions = [], []
    offset = 0
    for hemi, sign in (("left", -1), ("right", 1)):
        coords = load_surf_mesh(sphere[f"sphere_{hemi}"]).coordinates
        centers = coords[:PATCHES_PER_HEMISPHERE]
        assignment = cKDTree(centers).query(coords)[1]
        members.extend(np.flatnonzero(assignment == i) + offset for i in range(len(centers)))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        positions.append(np.column_stack([centers, np.full(len(centers), sign)]))
        offset += len(coords)
    # The sentinel points to a padded zero. Every real vertex occurs once.
    indices = np.full((len(members), max(map(len, members))), offset)
    for i, vertices in enumerate(members):
        indices[i, :len(vertices)] = vertices
    np.testing.assert_array_equal(np.sort(indices[indices != offset]), np.arange(offset))
    return indices, np.concatenate(positions).astype(np.float32)


class CorticalTransformer(nn.Module):
    def __init__(self, patch_indices, positions):
        super().__init__()
        self.register_buffer("patch_indices", torch.as_tensor(patch_indices))
        self.register_buffer("positions", torch.as_tensor(positions))
        self.patch_projection = nn.Linear(3 * patch_indices.shape[1], WIDTH)
        self.position_projection = nn.Linear(positions.shape[1], WIDTH)
        self.token_norm = nn.LayerNorm(WIDTH)
        self.latents = nn.Parameter(torch.randn(LATENTS, WIDTH) * 0.1)
        self.cross_attention = nn.MultiheadAttention(WIDTH, HEADS, dropout=DROPOUT, batch_first=True)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, dim_feedforward=2 * WIDTH,
                                           dropout=DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, DEPTH, norm=nn.LayerNorm(WIDTH),
                                             enable_nested_tensor=False)
        self.regressor = nn.Linear(WIDTH, 1)

    def patch_values(self, responses):
        # Keep t=0, t=1, and t=2 in distinct positions in each patch vector.
        padded = F.pad(responses, (0, 1))
        return padded[:, :, self.patch_indices].transpose(1, 2).flatten(2)

    def forward(self, responses):
        tokens = self.patch_projection(self.patch_values(responses))
        tokens = self.token_norm(tokens + self.position_projection(self.positions))
        latents = self.latents.expand(len(responses), -1, -1)
        attended, _ = self.cross_attention(latents, tokens, tokens, need_weights=False)
        return self.regressor(self.encoder(latents + attended).mean(dim=1)).squeeze(-1)


def fit_transformer(train, evaluation, ratings, epochs, patches, seed):
    """Train once and return predictions at the requested epoch counts."""
    torch.manual_seed(seed)
    device = train.device
    model = CorticalTransformer(*patches).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    center = float(ratings.mean())
    scale = max(float(ratings.std()), 1e-7)
    target = torch.as_tensor((ratings - center) / scale, dtype=torch.float32, device=device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    predictions = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        for batch in torch.randperm(len(train), device=device).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = F.smooth_l1_loss(model(train[batch]), target[batch], beta=0.2)
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                # Evaluate in FP32 at every selected checkpoint.
                value = torch.cat([model(chunk) for chunk in evaluation.split(BATCH_SIZE)])
            predictions[epoch] = value.cpu().numpy().astype(np.float64) * scale + center
    return predictions
