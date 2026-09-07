"""Fine-tune Krea 2's VAE for RGBA while anchoring its original RGB latent space.

Only the VAE is loaded. Images use straight alpha, normalized to [-1, 1].
Latents saved by this program are RAW VAE latents, without diffusion scaling.
See README.md for commands, loss definitions, and verification limitations.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
from accelerate import Accelerator
from accelerate.utils import DistributedType, DistributedDataParallelKwargs, broadcast_object_list, set_seed
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file, save_file
from safetensors import safe_open


BASE_MODEL = "krea/Krea-2-Turbo"
BASE_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
IMAGE_SUFFIXES = {".png", ".webp", ".tif", ".tiff", ".avif", ".jpg", ".jpeg", ".bmp"}
RGB_SUFFIXES = IMAGE_SUFFIXES
TRAINING_FORMAT = 2


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vae(source, subfolder=None, revision=None):
    # Passing only the VAE component prevents downloading the DiT/text encoder.
    source = str(source)
    if Path(source).is_file():
        raise ValueError("Use a Diffusers VAE directory (config.json + weights), not a ComfyUI single file.")
    if subfolder is None:
        subfolder = "" if (Path(source) / "config.json").exists() else "vae"
    kwargs = {"subfolder": subfolder} if subfolder else {}
    if revision:
        kwargs["revision"] = revision
    config = AutoencoderKLQwenImage.load_config(source, **kwargs)
    if config.get("_class_name") != "AutoencoderKLQwenImage":
        raise ValueError(f"Expected AutoencoderKLQwenImage, got {config.get('_class_name')}")
    return AutoencoderKLQwenImage.from_pretrained(source, torch_dtype=torch.float32, **kwargs)


def convert_rgba(vae):
    """Preserve causal-convolution type/padding and every pretrained RGB tensor."""
    channels = vae.config.input_channels
    if channels == 4:
        return vae
    if channels != 3:
        raise ValueError(f"Expected 3 or 4 input channels, got {channels}")
    old = vae.state_dict()
    config = dict(vae.config)
    config["input_channels"] = 4  # QwenImage uses this for both encoder and decoder.
    rgba = AutoencoderKLQwenImage.from_config(config)
    expanded = rgba.state_dict()
    changed = {"encoder.conv_in.weight", "decoder.conv_out.weight", "decoder.conv_out.bias"}
    with torch.no_grad():
        for key, value in old.items():
            if key not in changed:
                if value.shape != expanded[key].shape:
                    raise ValueError(f"Unexpected model architecture change: {key}")
                expanded[key].copy_(value)
        expanded["encoder.conv_in.weight"].zero_()
        expanded["encoder.conv_in.weight"][:, :3].copy_(old["encoder.conv_in.weight"])
        expanded["decoder.conv_out.weight"].zero_()
        expanded["decoder.conv_out.weight"][:3].copy_(old["decoder.conv_out.weight"])
        expanded["decoder.conv_out.bias"].fill_(1.0)  # normalized alpha +1 = opaque
        expanded["decoder.conv_out.bias"][:3].copy_(old["decoder.conv_out.bias"])
    rgba.load_state_dict(expanded, strict=True)
    return rgba


def decode_raw(vae, z):
    """Single-frame decoder without Diffusers' inference-only output clamp.

    The standard decode clamps to [-1,1], which can kill training gradients for
    overshooting alpha values. Use the same decoder and first-frame caches, then
    clamp ONLY for validation/export. This is deliberately limited to T=1.
    """
    if z.ndim != 5 or z.shape[2] != 1:
        raise ValueError("This trainer supports independent images, with latent shape B,C,1,H,W.")
    vae.clear_cache()
    try:
        h = vae.post_quant_conv(z)
        return vae.decoder(h, feat_cache=vae._feat_map, feat_idx=[0])
    finally:
        vae.clear_cache()


def forward_train(vae, x, checkpointing=False):
    def run(inputs):
        vae.clear_cache()
        try:
            posterior = vae.encode(inputs.unsqueeze(2)).latent_dist
            # Compute exp/sample/KL in FP32 even when convolution uses autocast.
            mu, logvar = posterior.mean.float(), posterior.logvar.float()
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            y = decode_raw(vae, z.to(posterior.mean.dtype)).squeeze(2)
            return y, mu, logvar
        finally:
            vae.clear_cache()
    if checkpointing:
        # Built-in per-block QwenImage checkpointing is not implemented upstream.
        # Whole-forward recomputation resets caches and preserves sampling RNG.
        return checkpoint(run, x, use_reentrant=False, preserve_rng_state=True)
    return run(x)


def compatibility_forward(vae, rgb, reference_z, checkpointing=False, align_encoder=True, preserve_decoder=True):
    """Opaque encoder alignment + decoder replay at the TEACHER's latent.

    Decoding the student's own encoding would permit encoder/decoder co-drift.
    The detached reference latent anchors the original decoder's coordinates.
    """
    def run(rgb_input, z):
        vae.clear_cache()
        try:
            mu = logvar = y = None
            if align_encoder:
                opaque = torch.cat([rgb_input, torch.ones_like(rgb_input[:, :1])], dim=1)
                posterior = vae.encode(opaque.unsqueeze(2)).latent_dist
                mu, logvar = posterior.mean.float(), posterior.logvar.float()
            if preserve_decoder:
                y = decode_raw(vae, z).squeeze(2)
            return y, mu, logvar
        finally:
            vae.clear_cache()
    if checkpointing:
        return checkpoint(run, rgb, reference_z, use_reentrant=False, preserve_rng_state=True)
    return run(rgb, reference_z)


def check_reference(student, reference):
    if reference.config.input_channels != 3:
        raise ValueError("--reference-model must be the ORIGINAL three-channel RGB VAE used by your DiT.")
    # Architecture and the DiT's raw/normalized latent interface must remain fixed.
    for key in ("z_dim", "dim_mult", "temperal_downsample", "latents_mean", "latents_std"):
        if student.config.get(key) != reference.config.get(key):
            raise ValueError(f"Student/reference latent interface mismatch: {key}")


@torch.no_grad()
def reference_targets(reference, rgb, sample=True, decode=True):
    reference.clear_cache()
    try:
        x = rgb.to(dtype=next(reference.parameters()).dtype)
        posterior = reference.encode(x.unsqueeze(2)).latent_dist
        mu, logvar = posterior.mean.float(), posterior.logvar.float()
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        # Match raw decoder values during training. Public decode is their clamp.
        y = decode_raw(reference, z.to(posterior.mean.dtype)).squeeze(2).float().detach() if decode else None
        return z.detach(), y, mu.detach(), logvar.detach()
    finally:
        reference.clear_cache()


def reference_kl(mu, logvar, ref_mu, ref_logvar):
    """Mean KL(q_student || q_reference), evaluated in FP32 without variance floors."""
    mu, logvar = mu.float(), logvar.float()
    ref_mu, ref_logvar = ref_mu.detach().float(), ref_logvar.detach().float()
    delta = logvar - ref_logvar
    return 0.5 * ((mu - ref_mu).square() * (-ref_logvar).exp() + torch.expm1(delta) - delta).mean()


def compatibility_loss(pred, mu, logvar, target_rgb, ref_mu, ref_logvar, weights):
    parts = {}
    if weights["ref_kl"]:
        parts["ref_kl"] = reference_kl(mu, logvar, ref_mu, ref_logvar)
    if weights["rgb_distill"]:
        parts["rgb_distill"] = F.mse_loss(pred[:, :3].float() * 0.5, target_rgb.detach().float() * 0.5)
    if weights["opaque_alpha"]:
        parts["opaque_alpha"] = ((pred[:, 3:4].float() - 1) * 0.5).square().mean()
    return parts


class TrainableVAE(nn.Module):
    """A real forward() lets Accelerate/DeepSpeed manage the entire trainable graph."""
    def __init__(self, vae, checkpointing=False, align_encoder=True, preserve_decoder=True):
        super().__init__()
        self.vae, self.checkpointing = vae, checkpointing
        self.align_encoder, self.preserve_decoder = align_encoder, preserve_decoder

    def forward(self, x, replay_rgb=None, reference_z=None):
        # ZeRO mixed precision casts model weights without Accelerate native AMP.
        dtype = next(self.vae.parameters()).dtype
        result = forward_train(self.vae, x.to(dtype=dtype), self.checkpointing)
        if replay_rgb is not None:
            result += compatibility_forward(self.vae, replay_rgb.to(dtype=dtype),
                                            reference_z.to(dtype=dtype), self.checkpointing,
                                            self.align_encoder, self.preserve_decoder)
        return result


class FullUpdateSampler(Sampler):
    """Shuffle once per epoch; pad to complete global accumulation groups.

    Padding repeats a shuffled prefix, avoiding short microbatches and partial
    ZeRO accumulation boundaries. Each rank gets equal full batches after shard.
    """
    def __init__(self, size, global_update_size, seed):
        self.size, self.seed, self.epoch = size, seed, 0
        self.count = math.ceil(size / global_update_size) * global_update_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.count

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        return iter((order * math.ceil(self.count / self.size))[:self.count])


def read_rgba(path, allow_rgb=False):
    if Path(path).suffix.lower() == ".avif" and ".avif" not in Image.registered_extensions():
        raise RuntimeError("AVIF decoding is unavailable. Install a Pillow >=11.3 wheel with AVIF support.")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" not in image.getbands() and "transparency" not in image.info and not allow_rgb:
            raise ValueError(f"Missing alpha channel: {path}. Use --allow-rgb only for intentional RGB mixing.")
        return image.convert("RGBA")


def to_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)


def list_images(root, suffixes=IMAGE_SUFFIXES):
    root = Path(root).resolve()
    files = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError(f"No supported images in {root}; expected {sorted(suffixes)}")
    return files


class RGBAImages(Dataset):
    def __init__(self, files, resolution, augment=False, seed=0, epoch=0, allow_rgb=False):
        self.files, self.resolution = files, resolution
        self.augment, self.seed, self.epoch, self.allow_rgb = augment, seed, epoch, allow_rgb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = read_rgba(self.files[index], self.allow_rgb)
        size = self.resolution
        scale = size / max(image.size)
        wh = tuple(max(1, round(v * scale)) for v in image.size)
        # Premultiplied resizing avoids dark fringes in semi-transparent pixels.
        image = image.convert("RGBa").resize(wh, Image.Resampling.LANCZOS).convert("RGBA")
        # RGB/fully opaque examples must stay opaque, including letterbox padding.
        opaque = image.getextrema()[3] == (255, 255)
        canvas = Image.new("RGBA", (size, size), (127, 127, 127, 255) if opaque else (0, 0, 0, 0))
        canvas.paste(image, ((size - wh[0]) // 2, (size - wh[1]) // 2))
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        if self.augment and rng.random() < 0.5:
            canvas = ImageOps.mirror(canvas)
        return to_tensor(canvas)


class CompatibilityImages(RGBAImages):
    """One RGBA example plus one opaque RGB example, reproducible across resume.

    Half of replay examples (when available) come from an additional RGB corpus;
    the rest are black/white composites, sampling AlphaVAE's opaque reference KL.
    Never equate a transparent foreground posterior with two different backgrounds.
    """
    def __init__(self, *args, replay_files=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_files = replay_files or []

    def __getitem__(self, index):
        rgba = super().__getitem__(index)
        rng = random.Random(self.seed + 91_771 + self.epoch * len(self) + index)
        bg = float(rng.randrange(2))
        if self.replay_files and rng.random() < 0.5:
            image = read_rgba(self.replay_files[rng.randrange(len(self.replay_files))], allow_rgb=True)
            image = image.convert("RGBa")
            image = ImageOps.fit(image, (self.resolution, self.resolution), method=Image.Resampling.LANCZOS)
            image = image.convert("RGBA")
            if self.augment and rng.random() < 0.5:
                image = ImageOps.mirror(image)
            source = to_tensor(image)
        else:
            source = rgba
        rgb = composite(source.unsqueeze(0), bg)[0] * 2 - 1
        return {"rgba": rgba, "replay_rgb": rgb}


def split_channels(x):
    rgb, alpha = (x[:, :3].float() + 1) / 2, (x[:, 3:4].float() + 1) / 2
    return rgb, alpha


def composite(x, background):
    rgb, alpha = split_channels(x)
    return rgb * alpha + background * (1 - alpha)


def abmse(pred, target):
    """Exact E_b ||A(pred,b)-A(target,b)||², b~Uniform([0,1]^3).

    E[b]=1/2 and Var[b]=1/12. The square-plus-variance form avoids cancellation.
    Mean reduction over batch, RGB channels and pixels; not a literal reproduction
    of AlphaVAE's ImageNet-estimated background moments and sum reduction.
    """
    rgb, a = split_channels(pred)
    gt_rgb, gt_a = split_channels(target)
    p, da = rgb * a - gt_rgb * gt_a, a - gt_a
    return ((p - 0.5 * da).square() + da.square() / 12).mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, lpips_weight=0.5, kl_weight=1e-6, alpha_weight=0.0):
        super().__init__()
        self.lpips_weight, self.kl_weight, self.alpha_weight = lpips_weight, kl_weight, alpha_weight
        self.perceptual = None
        if lpips_weight:
            import lpips
            self.perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False)

    def forward(self, pred, target, mu, logvar):
        reconstruction = abmse(pred, target)
        perceptual = reconstruction.new_zeros(())
        if self.perceptual is not None:
            # Both backgrounds share the same scale [0,1] before LPIPS [-1,1].
            for bg in (0.0, 1.0):
                p = composite(pred, bg) * 2 - 1
                t = composite(target, bg) * 2 - 1
                perceptual = perceptual + self.perceptual(p, t).mean() * 0.5
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
        alpha_l1 = ((pred[:, 3:4].float() - target[:, 3:4].float()) * 0.5).abs().mean()
        loss = reconstruction + self.lpips_weight * perceptual + self.kl_weight * kl + self.alpha_weight * alpha_l1
        return loss, {"abmse": reconstruction, "lpips": perceptual, "kl": kl, "alpha_l1": alpha_l1}


def autocast_context(device, precision):
    if precision == "no":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def select_device(name, precision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if precision != "no" and device.type != "cuda":
        raise ValueError("Use --precision no on CPU.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use --precision fp16 or no.")
    return device


def save_image(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)


@torch.no_grad()
def validate(vae, files, args, device, step):
    was_training = vae.training
    vae.eval()
    dataset = RGBAImages(files[:args.val_samples], args.resolution, allow_rgb=args.allow_rgb)
    total_mse, total_alpha = 0.0, 0.0
    backgrounds = ((0,0,0), (.5,.5,.5), (1,1,1), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1))
    try:
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)
            with autocast_context(device, args.precision):
                z = vae.encode(x.unsqueeze(2)).latent_dist.mode()
                y = vae.decode(z).sample.squeeze(2).float().clamp(-1, 1)
            mse = 0.0
            for bg in backgrounds:
                b = torch.tensor(bg, device=device).view(1, 3, 1, 1)
                mse += F.mse_loss(composite(y, b), composite(x, b)).item() / len(backgrounds)
            total_mse += mse
            total_alpha += ((y[:, 3:] - x[:, 3:]) / 2).abs().mean().item()
            if i < 4:
                folder = Path(args.output) / "validation" / f"step-{step:07d}"
                save_image(x[0], folder / f"{i:02d}-input.png")
                save_image(y[0], folder / f"{i:02d}-reconstruction.png")
        avg = total_mse / len(dataset)
        return {"val_composite_mse": avg, "val_composite_psnr": -10 * math.log10(max(avg, 1e-12)),
                "val_alpha_mae": total_alpha / len(dataset)}
    finally:
        vae.train(was_training)


@torch.no_grad()
def validate_compatibility(vae, reference, files, args, device, step, replay_files=None):
    """Held-out encoder/decoder compatibility proxy; not a DiT generation test."""
    was_training = vae.training
    vae.eval()
    dataset = CompatibilityImages(files[:args.val_samples], args.resolution, seed=args.seed,
                                  allow_rgb=args.allow_rgb, replay_files=replay_files)
    mse, opacity, kl = 0.0, 0.0, 0.0
    try:
        for i in range(len(dataset)):
            rgb = dataset[i]["replay_rgb"].unsqueeze(0).to(device)
            with autocast_context(device, args.precision):
                z, teacher_rgb, ref_mu, ref_logvar = reference_targets(reference, rgb, sample=False)
                dtype = next(vae.parameters()).dtype
                decoded, mu, logvar = compatibility_forward(vae, rgb.to(dtype), z.to(dtype))
            student = decoded.float().clamp(-1, 1)
            teacher_rgb = teacher_rgb.clamp(-1, 1)
            mse += F.mse_loss(student[:, :3] * 0.5, teacher_rgb * 0.5).item()
            opacity += ((student[:, 3:4] - 1) * 0.5).abs().mean().item()
            kl += reference_kl(mu, logvar, ref_mu, ref_logvar).item()
            if i < 4:
                folder = Path(args.output) / "validation" / f"step-{step:07d}"
                save_image(teacher_rgb[0], folder / f"{i:02d}-old-latent-teacher-rgb.png")
                save_image(student[0, :3], folder / f"{i:02d}-old-latent-student-rgb.png")
                save_image(student[0], folder / f"{i:02d}-old-latent-student-rgba.png")
        mse /= len(dataset)
        return {"compat_rgb_mse": mse, "compat_rgb_psnr": -10 * math.log10(max(mse, 1e-12)),
                "compat_alpha_mae": opacity / len(dataset), "compat_ref_kl": kl / len(dataset)}
    finally:
        vae.train(was_training)


def lr_schedule(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def export_vae(accelerator, model, folder):
    # ZeRO-2 replicates parameters; save_state below preserves FP32 optimizer masters.
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        vae_state = {k.removeprefix("vae."): v for k, v in state.items()}
        accelerator.unwrap_model(model).vae.save_pretrained(folder, state_dict=vae_state, safe_serialization=True)
    accelerator.wait_for_everyone()


def save_checkpoint(accelerator, model, args, step, epoch, next_batch, best):
    folder = Path(args.output) / f"checkpoint-{step:07d}"
    # All ranks must participate: ZeRO optimizer shards and per-rank RNG are needed.
    accelerator.save_state(str(folder / "state"))
    export_vae(accelerator, model, folder / "vae")
    if accelerator.is_main_process:
        write_json(folder / "progress.json", {"step": step, "epoch": epoch, "next_batch": next_batch, "best": best})
        write_json(folder / "training_args.json", vars(args))
    accelerator.wait_for_everyone()
    accelerator.print(f"Saved {folder}", flush=True)


def train(args):
    args.training_format = TRAINING_FORMAT
    weights = {"ref_kl": args.ref_kl_weight, "rgb_distill": args.rgb_distill_weight,
               "opaque_alpha": args.opaque_alpha_weight}
    active_compat = args.compatibility and any(weights.values())
    need_reference = args.compatibility and (active_compat or args.compat_validation)
    align_encoder = active_compat and weights["ref_kl"] > 0
    preserve_decoder = active_compat and (weights["rgb_distill"] > 0 or weights["opaque_alpha"] > 0)
    accelerator = Accelerator(
        mixed_precision=args.precision, cpu=args.device == "cpu",
        gradient_accumulation_steps=args.grad_accum, step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    args.precision = accelerator.mixed_precision
    device = accelerator.device
    args.world_size = accelerator.num_processes
    args.backend = str(accelerator.distributed_type)
    ds = accelerator.state.deepspeed_plugin
    if ds:
        config = ds.deepspeed_config
        if config.get("zero_optimization", {}).get("stage") != 2:
            raise ValueError("This training script targets DeepSpeed ZeRO-2.")
        if "optimizer" in config or "scheduler" in config:
            raise ValueError("Remove optimizer/scheduler sections from the DeepSpeed JSON: this script supplies AdamW + cosine.")
        for key, expected in {"gradient_accumulation_steps": args.grad_accum,
                              "train_micro_batch_size_per_gpu": args.batch_size,
                              "train_batch_size": args.batch_size * args.grad_accum * args.world_size,
                              "gradient_clipping": args.max_grad_norm}.items():
            actual = config.get(key, "auto")
            if actual != "auto" and actual != expected:
                raise ValueError(f"DeepSpeed {key}={actual} conflicts with script value {expected}; use auto or match it.")
            config[key] = expected
        if config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none") != "none":
            raise ValueError("Use ZeRO-2 without optimizer offload for this script's PyTorch cosine scheduler.")
        # Single-frame processing intentionally skips temporal-only convolutions.
        config["zero_optimization"]["ignore_unused_parameters"] = True
        args.deepspeed_config = json.loads(json.dumps(config))
    if device.type == "cpu" and args.precision != "no":
        raise ValueError("Use --precision no on CPU.")
    if args.resolution < 32 or args.resolution % 8:
        raise ValueError("--resolution must be >=32 and divisible by 8.")
    set_seed(args.seed)
    files = list_images(args.train_dir)
    if args.val_dir:
        val_files = list_images(args.val_dir)
        if set(files) & set(val_files):
            raise ValueError("Train and validation files overlap (including nested directories).")
    else:
        if len(files) < 2:
            raise ValueError("Need at least 2 images, or provide a separate --val-dir.")
        random.Random(args.seed).shuffle(files)
        count = max(1, min(len(files) - 1, round(len(files) * args.val_fraction)))
        val_files, files = files[:count], files[count:]
    replay_files, replay_val_files = [], []
    if need_reference and args.rgb_replay and args.rgb_replay_dir:
        replay_files = list_images(args.rgb_replay_dir, RGB_SUFFIXES)
        if args.rgb_replay_val_dir:
            replay_val_files = list_images(args.rgb_replay_val_dir, RGB_SUFFIXES)
        else:
            if len(replay_files) < 2:
                raise ValueError("Need >=2 RGB replay images for a hold-out split, or --rgb-replay-val-dir.")
            random.Random(args.seed).shuffle(replay_files)
            count = max(1, min(len(replay_files)-1, round(len(replay_files)*args.val_fraction)))
            replay_val_files, replay_files = replay_files[:count], replay_files[count:]
        if (set(files) | set(replay_files)) & (set(val_files) | set(replay_val_files)):
            raise ValueError("Main/replay training files overlap validation files. Keep the splits separate.")
    manifest = {"train": [str(p) for p in files], "validation": [str(p) for p in val_files],
                "rgb_replay_train": [str(p) for p in replay_files], "rgb_replay_validation": [str(p) for p in replay_val_files]}
    fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files + val_files + replay_files + replay_val_files]
    args.data_fingerprint = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Choose a new directory or use --resume.")
    reference = None
    if need_reference:
        args.reference_model = args.reference_model or args.model
        if args.reference_subfolder is None and args.reference_model == args.model:
            args.reference_subfolder = args.subfolder
        if args.reference_revision is None:
            args.reference_revision = (args.revision if args.reference_model == args.model else None)
            if args.reference_revision is None and args.reference_model == BASE_MODEL:
                args.reference_revision = BASE_REVISION
        with accelerator.main_process_first():
            reference = load_vae(args.reference_model, args.reference_subfolder, args.reference_revision).eval().requires_grad_(False)
        if reference.config.input_channels != 3:
            raise ValueError("Supply --reference-model pointing to the ORIGINAL RGB VAE, not an RGBA checkpoint.")
        digest = hashlib.sha256()
        for name, tensor in reference.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        args.reference_fingerprint = digest.hexdigest()
    progress = None
    if args.resume:
        saved_args = json.loads((Path(args.resume) / "training_args.json").read_text())
        if saved_args.get("training_format") != TRAINING_FORMAT:
            raise ValueError("Old training format: start a new run with --model OLD/vae and --reference-model ORIGINAL_RGB_VAE.")
        may_change = {"resume", "output", "device", "workers", "log_every", "save_every", "val_every", "val_samples"}
        for key, value in saved_args.items():
            if key not in may_change and getattr(args, key, None) != value:
                raise ValueError(f"Resume argument {key} changed: {value!r} -> {getattr(args, key, None)!r}")
        progress = json.loads((Path(args.resume) / "progress.json").read_text())
        vae = load_vae(Path(args.resume) / "vae")
    else:
        revision = args.revision or (BASE_REVISION if args.model == BASE_MODEL else None)
        with accelerator.main_process_first():
            vae = convert_rgba(load_vae(args.model, args.subfolder, revision))
    if vae.config.input_channels != 4:
        raise ValueError("RGBA training requires a four-channel VAE.")
    if reference is not None:
        check_reference(vae, reference)
        reference.to(device)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "training_args.json", vars(args))
        write_json(output / "split.json", manifest)
    accelerator.wait_for_everyone()
    model = TrainableVAE(vae.train().requires_grad_(True), args.checkpointing, align_encoder, preserve_decoder)
    criterion = ReconstructionLoss(args.lpips_weight, args.kl_weight, args.alpha_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    warmup = round(args.max_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_schedule(n, args.max_steps, warmup))
    dataset = (CompatibilityImages(files, args.resolution, args.flip, args.seed, allow_rgb=args.allow_rgb,
                                   replay_files=replay_files) if active_compat else
               RGBAImages(files, args.resolution, args.flip, args.seed, allow_rgb=args.allow_rgb))
    sampler = FullUpdateSampler(len(dataset), args.batch_size * args.grad_accum * args.world_size, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if accelerator.gradient_accumulation_steps != args.grad_accum or len(loader) % args.grad_accum:
        raise ValueError("Accumulation / loader alignment changed during Accelerate preparation.")
    step, epoch, skip, best = 0, 0, 0, math.inf
    if progress:
        accelerator.load_state(str(Path(args.resume) / "state"))
        step, epoch, skip, best = (progress[k] for k in ("step", "epoch", "next_batch", "best"))
    else:
        set_seed(args.seed + accelerator.process_index)
    accelerator.print(f"Backend={accelerator.distributed_type}, precision={args.precision}, world={args.world_size}, "
                      f"train={len(files)}, val={len(val_files)}, effective_batch={args.batch_size*args.grad_accum*args.world_size}, "
                      f"epoch_padding={len(sampler)-len(dataset)}", flush=True)
    accelerator.print(f"Compatibility={args.compatibility}, active_losses={weights if active_compat else {}}, "
                      f"RGB_replay={len(replay_files)}, mixed_RGB={args.allow_rgb}", flush=True)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        loader.set_epoch(epoch)
        epoch_loader = accelerator.skip_first_batches(loader, skip) if skip else loader
        logged = {}
        for index, batch in enumerate(epoch_loader, start=skip):
            x = batch["rgba"] if active_compat else batch
            with accelerator.accumulate(model):
                if active_compat:
                    with autocast_context(device, args.precision):
                        ref_z, ref_rgb, ref_mu, ref_logvar = reference_targets(
                            reference, batch["replay_rgb"], sample=preserve_decoder,
                            decode=weights["rgb_distill"] > 0)
                with accelerator.autocast():
                    result = model(x, batch["replay_rgb"], ref_z) if active_compat else model(x)
                    pred, mu, logvar = result[:3]
                # Frozen LPIPS and all loss math remain FP32, outside autocast.
                loss, parts = criterion(pred.float(), x.float(), mu.float(), logvar.float())
                if active_compat:
                    compat_parts = compatibility_loss(*result[3:], ref_rgb, ref_mu, ref_logvar, weights)
                    loss = loss + sum(weights[key] * value for key, value in compat_parts.items())
                    parts.update(compat_parts)
                finite = accelerator.reduce(torch.isfinite(loss.detach()).int(), reduction="sum")
                if finite.item() != accelerator.num_processes:
                    raise FloatingPointError(f"Non-finite loss at step {step} on at least one rank.")
                # Accelerate / DeepSpeed divides by accumulation steps exactly once.
                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                for key, value in {"loss": loss, **parts}.items():
                    logged[key] = logged.get(key, 0.0) + value.detach().item() / args.grad_accum
            if not accelerator.sync_gradients:
                continue
            if accelerator.optimizer_step_was_skipped:
                accelerator.print("AMP overflow: update skipped; LR and successful step count unchanged.")
                logged = {}
                continue
            scheduler.step()  # one cosine step per optimizer update, independent of world size
            step += 1
            next_epoch, next_batch = (epoch + 1, 0) if index + 1 == len(loader) else (epoch, index + 1)
            keys = list(logged)
            means = accelerator.reduce(torch.tensor([logged[k] for k in keys], device=device), reduction="mean")
            record = {"step": step, "epoch": epoch, "lr_next": scheduler.get_last_lr()[0],
                      **dict(zip(keys, means.cpu().tolist()))}
            if step % args.val_every == 0 or step == args.max_steps:
                metrics = [None]
                if accelerator.is_main_process:
                    metrics[0] = validate(accelerator.unwrap_model(model).vae, val_files, args, device, step)
                    if args.compatibility and args.compat_validation:
                        metrics[0].update(validate_compatibility(accelerator.unwrap_model(model).vae, reference,
                                                               val_files, args, device, step, replay_val_files))
                broadcast_object_list(metrics)
                record.update(metrics[0])
                if record["val_composite_mse"] < best:
                    best = record["val_composite_mse"]
                    export_vae(accelerator, model, output / "best" / "vae")
                    if accelerator.is_main_process:
                        write_json(output / "best" / "metrics.json", record)
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if step % args.log_every == 0 or step == 1 or step == args.max_steps:
                accelerator.print(json.dumps(record) + f" elapsed={time.time()-started:.1f}s", flush=True)
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(accelerator, model, args, step, next_epoch, next_batch, best)
            logged = {}
            if step >= args.max_steps:
                break
        epoch, skip = epoch + 1, 0
    export_vae(accelerator, model, output / "final" / "vae")
    accelerator.print(f"Finished: {output / 'final' / 'vae'}", flush=True)
    accelerator.end_training()


@torch.no_grad()
def inference(args):
    device = select_device(args.device, args.precision)
    vae = load_vae(args.model, args.subfolder, args.revision).to(device).eval()
    if vae.config.input_channels != 4:
        raise ValueError("Load a trained RGBA VAE, e.g. RUN/final/vae.")
    if args.tiling:
        vae.enable_tiling()
    if args.command == "decode":
        z = load_file(args.input, device="cpu")["latent"].to(device)
        with safe_open(args.input, framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        if meta.get("format") != "raw-qwen-rgba-vae-v1":
            raise ValueError("Expected raw latent file created by the encode command.")
        h, w = int(meta["height"]), int(meta["width"])
    else:
        x = to_tensor(read_rgba(args.input, allow_rgb=args.allow_rgb)).unsqueeze(0).to(device)
        h, w = x.shape[-2:]
        factor = 2 ** (len(vae.config.dim_mult) - 1)
        opaque = bool((x[:, 3:] == 1).all())
        x = F.pad(x, (0, (-w) % factor, 0, (-h) % factor), value=-1)
        if opaque:
            x[:, 3:] = 1
        with autocast_context(device, args.precision):
            z = vae.encode(x.unsqueeze(2)).latent_dist.mode()
    if args.command == "encode":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_file({"latent": z.float().cpu().contiguous()}, args.output,
                  metadata={"format": "raw-qwen-rgba-vae-v1", "height": str(h), "width": str(w), "model": args.model})
    else:
        with autocast_context(device, args.precision):
            y = vae.decode(z).sample[0, :, 0, :h, :w]
        save_image(y[:3] if args.rgb_output else y, args.output)
    print(f"Saved {args.output}")


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Convert RGB VAE and fine-tune all weights for RGBA reconstruction.")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--subfolder", default=None, help="Auto-detect local VAE folder; otherwise defaults to vae.")
    p.add_argument("--revision", default=None)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Checkpoint directory, NOT its vae subdirectory.")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=32000, help="Optimizer updates, not microbatches.")
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--lpips-weight", type=float, default=0.5)
    p.add_argument("--alpha-weight", type=float, default=0.0, help="Optional extra alpha L1, off by default.")
    p.add_argument("--compatibility", action=argparse.BooleanOptionalAction, default=True,
                   help="RGB compatibility master switch; --no-compatibility skips the reference VAE entirely.")
    p.add_argument("--ref-kl-weight", type=float, default=1e-3, help="Opaque encoder reference KL; 0 disables its branch.")
    p.add_argument("--rgb-distill-weight", type=float, default=1.0, help="RGB decoder distillation at old latents; 0 disables.")
    p.add_argument("--opaque-alpha-weight", type=float, default=0.1, help="Alpha=1 at old RGB latents; 0 disables.")
    p.add_argument("--compat-validation", action=argparse.BooleanOptionalAction, default=True,
                   help="Held-out old-latent compatibility metrics; requires the master switch.")
    p.add_argument("--reference-model", help="Original RGB VAE used by the DiT; defaults to --model.")
    p.add_argument("--reference-subfolder")
    p.add_argument("--reference-revision")
    p.add_argument("--rgb-replay-dir", help="Optional extra RGB replay corpus. Main train-dir can already mix RGB/RGBA.")
    p.add_argument("--rgb-replay", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable optional extra corpus if a directory is supplied; --no-rgb-replay ignores it.")
    p.add_argument("--rgb-replay-val-dir", help="Optional held-out replay corpus; otherwise split from rgb-replay-dir.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers per rank (Linux).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flip", action="store_true", help="Optional horizontal flips; leave off for text/icons.")
    p.add_argument("--allow-rgb", action=argparse.BooleanOptionalAction, default=True,
                   help="Accept mixed RGB/RGBA including AVIF; --no-allow-rgb requires an alpha channel.")
    p.add_argument("--checkpointing", action="store_true", help="Whole-forward activation recomputation.")
    for name in ("reconstruct", "encode", "decode"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--subfolder", default=None)
        p.add_argument("--revision", default=None)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--tiling", action="store_true", help="Inference only; may change border reconstruction.")
        p.add_argument("--allow-rgb", action=argparse.BooleanOptionalAction, default=True)
        if name != "encode":
            p.add_argument("--rgb-output", action="store_true", help="Save RGB channels only, ignoring alpha (no compositing).")
    for p in commands.choices.values():
        p.add_argument("--device", default="auto")
        p.add_argument("--precision", choices=("no", "bf16", "fp16"), default="no")
    commands.choices["train"].set_defaults(precision=None)
    return root


def main():
    args = parser().parse_args()
    if args.command == "train":
        for key in ("batch_size", "grad_accum", "max_steps", "val_samples", "val_every", "save_every", "log_every"):
            if getattr(args, key) < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be positive.")
        if not 0 < args.val_fraction < 1 or not 0 <= args.warmup_ratio < 1:
            raise ValueError("Invalid validation fraction or warmup ratio.")
        if args.lr <= 0 or args.workers < 0 or args.max_grad_norm <= 0:
            raise ValueError("Invalid LR, workers, or gradient clipping setting.")
        if min(args.kl_weight, args.lpips_weight, args.alpha_weight, args.weight_decay,
               args.ref_kl_weight, args.rgb_distill_weight, args.opaque_alpha_weight) < 0:
            raise ValueError("Loss weights and weight decay must be nonnegative.")
        if args.rgb_replay_val_dir and not args.rgb_replay_dir:
            raise ValueError("--rgb-replay-val-dir requires --rgb-replay-dir.")
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
