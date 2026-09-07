"""Explicit Krea2Pipeline packed/normalized latent <-> raw VAE interface.

No DiT is loaded here. This matches the official Diffusers Krea2Pipeline packing
order and its z_raw = z_normalized * latents_std + latents_mean convention.
Other pipelines/backends must use their own verified latent conversion.
"""
import torch


def latent_stats(vae, tensor):
    channels = vae.config.z_dim
    mean = torch.as_tensor(vae.config.latents_mean, device=tensor.device, dtype=torch.float32)
    std = torch.as_tensor(vae.config.latents_std, device=tensor.device, dtype=torch.float32)
    if mean.numel() != channels or std.numel() != channels or not bool(torch.all(std > 0)):
        raise ValueError("Invalid original VAE latent statistics.")
    return mean.view(1, channels, 1, 1, 1), std.view(1, channels, 1, 1, 1)


def pack_krea_latents(raw, vae, patch_size=2):
    """For downstream DiT training: [B,C,1,h,w] raw -> [B,N,C*p*p] normalized."""
    if raw.ndim != 5 or raw.shape[1] != vae.config.z_dim or raw.shape[2] != 1:
        raise ValueError("Expected raw latent [B,z_dim,1,h,w].")
    batch, channels, _, h, w = raw.shape
    p = patch_size
    if p < 1 or h % p or w % p:
        raise ValueError("Latent spatial dimensions must be divisible by patch_size.")
    mean, std = latent_stats(vae, raw)
    normalized = (raw.float() - mean) / std
    return (normalized[:, :, 0].reshape(batch, channels, h // p, p, w // p, p)
            .permute(0, 2, 4, 1, 3, 5).reshape(batch, (h // p) * (w // p), channels * p * p))


def unpack_krea_latents(packed, vae, height, width, patch_size=2):
    """Krea2Pipeline(output_type='latent').images -> RAW latent for VAE decode.

    height/width are the actual generated pixel dimensions (after any rounding).
    This accepts the Diffusers packed layout only, not ComfyUI's latent object.
    """
    scale = 2 ** (len(vae.config.dim_mult) - 1)
    p = patch_size
    if p < 1 or min(height, width) < 1 or height % (scale * p) or width % (scale * p):
        raise ValueError("Pixel dimensions must be positive multiples of VAE_scale * patch_size.")
    h, w, channels = height // scale, width // scale, vae.config.z_dim
    if packed.ndim != 3 or tuple(packed.shape[1:]) != ((h // p) * (w // p), channels * p * p):
        raise ValueError("Packed latent shape does not match dimensions, z_dim, and patch_size.")
    batch = packed.shape[0]
    normalized = (packed.float().reshape(batch, h // p, w // p, channels, p, p)
                  .permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, 1, h, w))
    mean, std = latent_stats(vae, normalized)
    return normalized * std + mean


@torch.no_grad()
def decode_krea_latents(vae, packed, height, width, rgb_only=False, patch_size=2):
    """Return [B,3/4,H,W] in [0,1]. rgb_only discards alpha without compositing."""
    param = next(vae.parameters())
    raw = unpack_krea_latents(packed.to(param.device), vae, height, width, patch_size)
    image = vae.decode(raw.to(param.dtype)).sample[:, :, 0].float()
    if rgb_only:
        image = image[:, :3]
    return ((image + 1) / 2).clamp(0, 1)
