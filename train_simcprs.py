import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

from dataset_loader import MedPRSDatasetLoader
from stage1_retriever import BioBERTSimCPSREncoder

class MedPRSTrainDataset(Dataset):
    def __init__(self, papers_df, tokenizer, max_length=512):
        self.papers = papers_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        row = self.papers.iloc[idx]
        title = str(row.get('Title', ''))
        abstract = str(row.get('Abstract', ''))
        keywords = str(row.get('Keywords', ''))
        label = int(row.get('Label', 0))

        text = f"Title: {title}. Abstract: {abstract}. Keywords: {keywords}"
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def train_overnight(data_dir="/kaggle/input/medprs-dataset/", output_dir="/kaggle/working/", epochs=5, batch_size=32, lr=5e-5):
    print("=======================================================")
    print("STARTING OVERNIGHT TRAINING: BioBERT SimCPSR (MedPRS)")
    print(f"Data Dir: {data_dir} | Output Dir: {output_dir}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=======================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")

    # 1. Load Datasets
    loader = MedPRSDatasetLoader(data_dir=data_dir)
    journal_df = loader.load_journals()
    num_classes = len(journal_df)
    print(f"[Trainer] Number of target journal classes: {num_classes}")

    train_df = loader.load_papers(split="train")
    val_df = loader.load_papers(split="val")

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    train_dataset = MedPRSTrainDataset(train_df, tokenizer)
    val_dataset = MedPRSTrainDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Model & Softmax Classifier Head
    model = BioBERTSimCPSREncoder().to(device)
    classifier_head = nn.Linear(768, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier_head.parameters()), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.train()
        classifier_head.train()

        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            embeddings = model(input_ids, attention_mask)
            logits = classifier_head(embeddings)

            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 500 == 0 or step == len(train_loader):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {total_loss / step:.4f} | Elapsed: {elapsed/60:.1f}m")

        # Validation Phase
        print(f"\n[Validation] Evaluating Epoch {epoch} on Validation Set...")
        model.eval()
        classifier_head.eval()

        val_hits_1, val_hits_10, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                embeddings = model(input_ids, attention_mask)
                logits = classifier_head(embeddings)

                _, top1_preds = torch.max(logits, dim=1)
                val_hits_1 += (top1_preds == labels).sum().item()

                _, top10_preds = torch.topk(logits, k=10, dim=1)
                val_hits_10 += (top10_preds == labels.unsqueeze(1)).sum().item()

                val_total += labels.size(0)

        val_acc1 = (val_hits_1 / val_total) * 100.0
        val_acc10 = (val_hits_10 / val_total) * 100.0
        print(f"==> Epoch {epoch} Validation Top-1 Acc: {val_acc1:.2f}% | Top-10 Acc: {val_acc10:.2f}%")

        # Save Best Checkpoint
        checkpoint_save_path = os.path.join(output_dir, "best_simcprs_checkpoint.pth")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "classifier_head": classifier_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc10": val_acc10
        }

        if val_acc10 > best_val_acc:
            best_val_acc = val_acc10
            torch.save(save_dict, checkpoint_save_path)
            print(f"🎉 Saved NEW BEST Checkpoint to {checkpoint_save_path} (Val Top-10: {val_acc10:.2f}%)")

    print("\n=======================================================")
    print(f"OVERNIGHT TRAINING COMPLETED! Best Val Top-10 Acc: {best_val_acc:.2f}%")
    print(f"Saved Checkpoint File: {os.path.join(output_dir, 'best_simcprs_checkpoint.pth')}")
    print("=======================================================")

if __name__ == "__main__":
    train_overnight()
