#!/usr/bin/env python3
"""Train an instance segmentation model and save resumable checkpoints."""

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mambais.config import load_config
from mambais.loss import SetCriterion
from mambais.runtime import build, seed_everything, to_device


def main():
    parser = argparse.ArgumentParser(description="Train MambaIS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/mambais")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, loader = build(cfg, True)
    model.to(device)
    criterion = SetCriterion(len(cfg["data"]["class_names"]), **cfg["loss"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["train"]["amp"] and device.type == "cuda"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["epoch"] + 1
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
    for epoch in range(start, cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(batch)
                loss, parts = criterion(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"].get("grad_clip", 1.0)
            )
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % cfg["train"].get("log_every", 10) == 0:
                print(
                    json.dumps(
                        {"epoch": epoch, "step": step, "loss": loss.item(), **parts}
                    ),
                    flush=True,
                )
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, output / "last.pt")
        if (epoch + 1) % cfg["train"].get("save_every", 10) == 0:
            torch.save(checkpoint, output / f"epoch_{epoch+1:03d}.pt")
        print(f"epoch={epoch} mean_loss={running/max(1,len(loader)):.5f}", flush=True)


if __name__ == "__main__":
    main()
