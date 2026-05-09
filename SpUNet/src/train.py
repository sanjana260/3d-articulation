"""
Training script v2 — ArticulateSPUNet + DINOv2
===============================================
Key fix: part ID remapping — compact class space built from training data.
Checkpoints saved to checkpoints_v2/ (separate from previous runs).
"""

import os, sys, argparse, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT   = "/scratch/ky2751/articulate3d_project"
DATA_ROOT = f"{PROJECT}/downloadscript/scannet_full/data"
ANNO_ROOT = f"{PROJECT}/annotations"
CKPT_DIR  = f"{PROJECT}/checkpoints_v2"   # new folder, clean slate
LOG_DIR   = f"{PROJECT}/logs"
VIZ_DIR   = f"{PROJECT}/viz_v2"

sys.path.insert(0, f"{PROJECT}/Pointcept")
sys.path.insert(0, os.path.dirname(__file__))

from articulate3d_dataset import Articulate3DDataset, collate_fn, build_part_id_remap
from model import ArticulateSPUNet, ArticulationLoss


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, world_size, torch.device(f"cuda:{local_rank}")

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main(local_rank):
    return local_rank == 0


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",       action="store_true")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--max_points",  type=int,   default=150_000)
    p.add_argument("--voxel_size",  type=float, default=0.05)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--val_every",   type=int,   default=5)
    p.add_argument("--no_dino",     action="store_true")
    p.add_argument("--log_every",   type=int,   default=10)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ma(axis_pred, axis_gt, mob_gt, threshold_deg=11.25):
    mobile = mob_gt > 0
    if mobile.sum() == 0:
        return 0.0
    ap  = axis_pred[mobile]
    ag  = axis_gt[mobile]
    cos = (ap * ag).sum(-1).abs().clamp(0, 1)
    return (torch.acos(cos) * 180.0 / math.pi < threshold_deg).float().mean().item()

def compute_mo(origin_pred, origin_gt, mob_gt, threshold_m=0.1):
    mobile = mob_gt > 0
    if mobile.sum() == 0:
        return 0.0
    d = (origin_pred[mobile] - origin_gt[mobile]).norm(dim=-1)
    return (d < threshold_m).float().mean().item()

def compute_mob_acc(mob_logits, mob_gt):
    return (mob_logits.argmax(-1) == mob_gt).float().mean().item()

def compute_ap50_approx(seg_logits, part_gt):
    pred  = seg_logits.argmax(-1)
    parts = part_gt.unique()
    ious  = []
    for pid in parts:
        if pid == 0:
            continue
        pred_mask = pred == pid
        gt_mask   = part_gt == pid
        inter = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch": epoch, "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(), "metrics": metrics,
    }, path)
    print(f"  [ckpt] → {path}")

def load_ckpt(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    raw  = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  [ckpt] resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_viz(batch, preds, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    coords    = batch["coord"].cpu().numpy()
    mob_pred  = preds["mob_logits"].argmax(-1).cpu().numpy()
    mob_gt    = batch["mobility"].cpu().numpy()
    axis_pred = preds["axis_pred"].cpu().numpy()
    seg_pred  = preds["seg_logits"].argmax(-1).cpu().numpy()
    offset    = batch["offset"].cpu().numpy()
    start = 0
    for i, scene_id in enumerate(batch["scene_id"]):
        end = offset[i]
        np.savez_compressed(
            os.path.join(out_dir, f"ep{epoch:03d}_{scene_id}.npz"),
            coords=coords[start:end], mob_pred=mob_pred[start:end],
            mob_gt=mob_gt[start:end], axis_pred=axis_pred[start:end],
            seg_pred=seg_pred[start:end],
        )
        start = end


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_val(model, loader, loss_fn, device, epoch, world_size,
            save_viz_flag=False):
    model.eval()
    tot_loss = ma_sum = mo_sum = mob_sum = ap_sum = 0.0
    n = 0

    for batch_i, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        raw   = model.module if isinstance(model, DDP) else model
        preds = raw(batch)
        loss, _ = loss_fn(preds, batch)

        tot_loss += loss.item()
        ma_sum   += compute_ma(preds["axis_pred"], batch["axis"], batch["mobility"])
        mo_sum   += compute_mo(preds["origin_pred"], batch["origin"], batch["mobility"])
        mob_sum  += compute_mob_acc(preds["mob_logits"], batch["mobility"])
        ap_sum   += compute_ap50_approx(preds["seg_logits"], batch["part_id"])
        n += 1

        if save_viz_flag and batch_i < 3:
            save_viz(batch, preds, epoch, VIZ_DIR)

    metrics_t = torch.tensor(
        [tot_loss, ma_sum, mo_sum, mob_sum, ap_sum, n],
        dtype=torch.float64, device=device
    )
    if world_size > 1:
        dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)

    tot_loss, ma_sum, mo_sum, mob_sum, ap_sum, n = metrics_t.tolist()
    n = max(n, 1)
    model.train()
    return {
        "val_loss": tot_loss / n,
        "AP50":     ap_sum   / n,
        "MA":       ma_sum   / n,
        "MO":       mo_sum   / n,
        "mob_acc":  mob_sum  / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    local_rank, world_size, device = setup_ddp()
    main_proc = is_main(local_rank)

    if main_proc:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR,  exist_ok=True)
        os.makedirs(VIZ_DIR,  exist_ok=True)
        print(f"Device: {device}  |  world_size: {world_size}")

    if args.smoke:
        if main_proc:
            print("\n=== SMOKE TEST (2 epochs, 3 scenes) ===\n")
        args.epochs      = 2
        args.batch_size  = 1
        args.max_points  = 10_000
        args.num_workers = 0
        args.val_every   = 1

    # auto-resume from checkpoints_v2/latest.pth only
    latest = f"{CKPT_DIR}/latest.pth"
    if args.resume is None and os.path.exists(latest) and not args.smoke:
        args.resume = latest
        if main_proc:
            print(f"  [auto-resume] found {latest}")

    # ── build part ID remap from training scenes (main proc only, then used by all) ──
    train_split_file = os.path.join(ANNO_ROOT, "train.txt")
    with open(train_split_file) as f:
        all_train_ids = [l.strip() for l in f if l.strip()]
    train_scene_ids = [
        sid for sid in all_train_ids
        if os.path.exists(os.path.join(DATA_ROOT, sid, "scans", "mesh_aligned_0.05.ply"))
    ]
    part_id_remap, num_parts = build_part_id_remap(DATA_ROOT, ANNO_ROOT, train_scene_ids)
    if main_proc:
        print(f"[Remap] {num_parts - 1} unique foreground part IDs → compact 1..{num_parts-1}")
        print(f"[Remap] num_parts (incl background): {num_parts}")

    # ── datasets ─────────────────────────────────────────────────────────────
    train_ds = Articulate3DDataset(DATA_ROOT, ANNO_ROOT, "train",
                                   args.voxel_size, args.max_points,
                                   augment=not args.smoke,
                                   part_id_remap=part_id_remap)
    val_ds   = Articulate3DDataset(DATA_ROOT, ANNO_ROOT, "val",
                                   args.voxel_size, args.max_points,
                                   augment=False,
                                   part_id_remap=part_id_remap)

    if args.smoke:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(3, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(2, len(val_ds)))))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True,
                                       drop_last=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=local_rank, shuffle=False,
                                       drop_last=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              sampler=val_sampler,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))

    if main_proc:
        print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── model — use actual num_parts from remap ───────────────────────────────
    model   = ArticulateSPUNet(in_channels=6, num_parts=num_parts,
                               use_dino=not args.no_dino).to(device)
    loss_fn = ArticulationLoss(num_parts=num_parts)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if main_proc:
        print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_ckpt(model, optimizer, args.resume, device)
        for _ in range(start_epoch):
            scheduler.step()

    log_f = open(os.path.join(LOG_DIR, "train_log_v2.jsonl"), "a") if main_proc else None

    if main_proc:
        print(f"\nStarting training: {args.epochs} epochs\n")
    best_ap = 0.0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        ep_loss = 0.0
        t0 = time.time()

        for bi, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            preds = model(batch)
            loss, breakdown = loss_fn(preds, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_loss += loss.item()

            if main_proc and (bi + 1) % args.log_every == 0:
                print(f"  Ep {epoch+1:03d} | {bi+1:04d}/{len(train_loader)} "
                      f"| loss {loss.item():.4f} "
                      f"| seg {breakdown['L_seg']:.3f} "
                      f"| mob {breakdown['L_mob']:.3f} "
                      f"| axis {breakdown['L_axis']:.3f}")

        avg_loss = ep_loss / len(train_loader)
        elapsed  = time.time() - t0
        scheduler.step()

        if main_proc:
            print(f"\nEpoch {epoch+1:03d} | loss {avg_loss:.4f} | {elapsed:.0f}s "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")

        if main_proc and not args.smoke:
            save_ckpt(model, optimizer, epoch+1,
                      {"train_loss": avg_loss}, f"{CKPT_DIR}/latest.pth")
            if (epoch + 1) % 5 == 0:
                save_ckpt(model, optimizer, epoch+1,
                          {"train_loss": avg_loss},
                          f"{CKPT_DIR}/epoch_{epoch+1:03d}.pth")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            save_viz_now = (epoch + 1) % 10 == 0 or epoch == args.epochs - 1
            val_m = run_val(model, val_loader, loss_fn, device,
                            epoch+1, world_size, save_viz_flag=save_viz_now)

            if main_proc:
                print(f"  [VAL] loss {val_m['val_loss']:.4f} "
                      f"| AP50 {val_m['AP50']*100:.1f}% "
                      f"| MA {val_m['MA']*100:.1f}% "
                      f"| MO {val_m['MO']*100:.1f}% "
                      f"| mob_acc {val_m['mob_acc']*100:.1f}%")

                if val_m["AP50"] > best_ap and not args.smoke:
                    best_ap = val_m["AP50"]
                    save_ckpt(model, optimizer, epoch+1, val_m,
                              f"{CKPT_DIR}/best.pth")
                    print(f"  [best] New best AP50: {best_ap*100:.1f}%")

                log_entry = {"epoch": epoch+1, "train_loss": avg_loss, **val_m}
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()

    if main_proc:
        print("\nTraining complete.")
        if log_f:
            log_f.close()
        if args.smoke:
            print("\n✓ SMOKE TEST PASSED — ready to submit GPU job")

    cleanup_ddp()


if __name__ == "__main__":
    main()

