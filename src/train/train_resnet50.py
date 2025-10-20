import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
)
from torch.utils.data import DataLoader
from src.data.dataloaders import ManifestDataset


def parse_args():
    p = argparse.ArgumentParser()
    # core training/data
    p.add_argument("--csv", type=str,
                   default=os.getenv("CSV", "data/processed/figshare_manifest.csv"))
    p.add_argument("--batch", type=int, default=int(os.getenv("BATCH", 32)))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 25)))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", 3e-4)))
    p.add_argument("--num_classes", type=int, default=int(os.getenv("NUM_CLASSES", 3)))
    p.add_argument("--device", type=str,
                   default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)))
    # optional W&B
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "idl"))
    p.add_argument("--wandb_run_name", type=str, default=os.getenv("WANDB_RUN_NAME", "resnet50"))
    return p.parse_args()


def make_loaders(csv_path: str, batch_size: int):
    tr = DataLoader(ManifestDataset(csv_path, "train"), batch_size=batch_size,
                    shuffle=True, num_workers=2)
    va = DataLoader(ManifestDataset(csv_path, "val"), batch_size=batch_size,
                    shuffle=False, num_workers=2)
    te = DataLoader(ManifestDataset(csv_path, "test"), batch_size=batch_size,
                    shuffle=False, num_workers=2)
    return tr, va, te


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    acc = MulticlassAccuracy(num_classes=model.fc.out_features).to(device)
    f1 = MulticlassF1Score(num_classes=model.fc.out_features, average="macro").to(device)
    auc = MulticlassAUROC(num_classes=model.fc.out_features, average="macro").to(device)

    tot_loss = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * len(x)
        n += len(x)
        acc.update(logits, y)
        f1.update(logits, y)
        auc.update(logits, y)

    return (tot_loss / max(1, n),
            acc.compute().item(),
            f1.compute().item(),
            auc.compute().item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # optional W&B
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project,
                        name=args.wandb_run_name,
                        config={k: v for k, v in vars(args).items()
                                if k not in ["wandb"]})

    tr, va, te = make_loaders(args.csv, args.batch)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.to(args.device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ensure reports/ exists once up front
    os.makedirs("reports", exist_ok=True)

    best_f1 = -1.0
    best_state = None

    for e in range(1, args.epochs + 1):
        model.train()
        running = n = 0
        for x, y in tr:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
            n += len(x)

        vloss, vacc, vf1, vauc = evaluate(model, va, args.device)
        print(f"epoch {e:02d} | val loss {vloss:.4f} acc {vacc:.3f} f1 {vf1:.3f} auc {vauc:.3f}")

        if wb:
            wb.log({
                "epoch": e,
                "val/loss": vloss, "val/acc": vacc, "val/f1": vf1, "val/auroc": vauc,
                "train/loss": running / max(1, n)
            })

        # --- save LATEST every epoch ---
        last_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(last_state, "reports/last_resnet50.pt")

        # --- track BEST by validation F1 (save at end) ---
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = last_state  # already on CPU

    # save BEST and final test
    if best_state is None:
        # fallback in case no improvement recorded
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, "reports/best_resnet50.pt")

    model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
    tloss, tacc, tf1, tauc = evaluate(model, te, args.device)
    print(f"TEST | loss {tloss:.4f} acc {tacc:.3f} f1 {tf1:.3f} auc {tauc:.3f}")

    if wb:
        wb.log({"test/loss": tloss, "test/acc": tacc, "test/f1": tf1, "test/auroc": tauc})
        wb.finish()


if __name__ == "__main__":
    main()
