import yaml
import torch
import sys
import os
import json
from pathlib import Path

# Add src to path just in case
sys.path.append(str(Path(__file__).parent))

from data import build_loaders
from train import LSTMClassifier

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_smoke(config_path="configs/nlp_agnews.yaml"):
    print(f"Running smoke test with config: {config_path}")
    
    # 1. Load Config
    cfg = load_yaml(config_path)
    
    # Force num_workers=0 pour éviter les blocages sur le smoke test
    if 'data' in cfg:
        print("Note: Forcing num_workers=0 for smoke test to prevent hanging.")
        cfg['data']['num_workers'] = 0

    # 2. Build Loaders
    print("Building loaders and vocabulary... (Please wait, tokenizing large text can take ~30-60s)")
    train_loader, val_loader, test_loader, vocab, num_classes, label_names = build_loaders(cfg)
    
    print(f"Vocab size: {len(vocab.itos)}")
    print(f"Num classes: {num_classes}")

    # 3. Get one batch
    print("Fetching one batch...")
    batch = next(iter(train_loader))
    texts, lengths, labels = batch
    
    print(f"Batch shape: {texts.shape}")
    print(f"Labels shape: {labels.shape}")

    # 4. Init Model
    print("Initializing model...")
    
    # --- FIX ICI : Ajout de pad_idx ---
    model = LSTMClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        num_classes=num_classes,
        pad_idx=vocab.pad_idx  # <--- L'argument manquant qui causait l'erreur
    )

    # 5. Forward Pass
    print("Running forward pass...")
    # Move to CPU for smoke test
    texts = texts.to("cpu")
    lengths = lengths.to("cpu")
    model = model.to("cpu")
    
    logits = model(texts, lengths)
    
    # 6. Compute Loss (Dummy)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    
    print(f"Smoke test success! Loss: {loss.item()}")
    
    # Save dummy metrics
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "smoke_metrics.json"
    
    metrics = {
        "loss": loss.item(),
        "batch_size": texts.shape[0],
        "seq_len": texts.shape[1],
        "num_classes": num_classes
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics_path

if __name__ == "__main__":
    run_smoke()