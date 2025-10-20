import os
import torch, torch.nn as nn, torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
from torch.utils.data import DataLoader
from src.data.dataloaders import ManifestDataset

CSV="data/processed/figshare_manifest.csv"
BATCH=32
EPOCHS=25
LR=3e-4
NUM_CLASSES=3
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def loaders():
    tr = ManifestDataset(CSV,"train")
    va = ManifestDataset(CSV,"val")
    te = ManifestDataset(CSV,"test")

    # macOS often behaves better with workers=0; Colab can use >0
    workers = 0 if (os.uname().sysname == "Darwin" and DEVICE=="cpu") else 4
    pin = (DEVICE=="cuda")

    return (
        DataLoader(tr,batch_size=BATCH,shuffle=True ,num_workers=workers,pin_memory=pin),
        DataLoader(va,batch_size=BATCH,shuffle=False,num_workers=workers,pin_memory=pin),
        DataLoader(te,batch_size=BATCH,shuffle=False,num_workers=workers,pin_memory=pin),
    )

def evaluate(m, loader):
    m.eval()
    acc=MulticlassAccuracy(num_classes=NUM_CLASSES,average='macro').to(DEVICE)
    f1 =MulticlassF1Score (num_classes=NUM_CLASSES,average='macro').to(DEVICE)
    auc=MulticlassAUROC  (num_classes=NUM_CLASSES).to(DEVICE)
    loss_sum=0;n=0
    ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            logits=m(x); loss=ce(logits,y)
            loss_sum+=loss.item()*x.size(0); n+=x.size(0)
            prob=torch.softmax(logits,dim=1)
            acc.update(prob,y); f1.update(prob,y); auc.update(prob,y)
    return loss_sum/n, acc.compute().item(), f1.compute().item(), auc.compute().item()

def main():
    tr,va,te = loaders()
    m=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc=nn.Linear(m.fc.in_features,NUM_CLASSES)
    m=m.to(DEVICE)

    opt=optim.AdamW(m.parameters(),lr=LR)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    ce=nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1=-1; best=None
    os.makedirs("reports", exist_ok=True)

    for e in range(1, EPOCHS+1):
        m.train()
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            loss=ce(m(x),y)
            loss.backward(); opt.step()
        sch.step()

        vloss,vacc,vf1,vauc=evaluate(m,va)
        print(f"epoch {e:02d} | val loss {vloss:.4f} acc {vacc:.3f} f1 {vf1:.3f} auc {vauc:.3f}")
        if vf1>best_f1:
            best_f1=vf1
            best={k:v.cpu() for k,v in m.state_dict().items()}

    torch.save(best,"reports/best_resnet50.pt")
    m.load_state_dict({k:v.to(DEVICE) for k,v in best.items()})
    tloss,tacc,tf1,tauc = evaluate(m,te)
    print(f"TEST | loss {tloss:.4f} acc {tacc:.3f} f1 {tf1:.3f} auc {tauc:.3f}")

if __name__=="__main__":
    main()
