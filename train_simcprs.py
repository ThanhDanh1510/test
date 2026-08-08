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

def train_overnight(
    data_dir="/kaggle/input/medprs-dataset/",
    output_dir="/kaggle/working/",
    epochs=10, # Chuẩn 10 Epochs theo bài báo MedPRS gốc
    batch_size=64, # Optimized for 2x T4 GPUs (32 per GPU)
    lr=5e-5,
    save_step_frequency=1000, # Save checkpoint every 1000 steps
    use_fp16=True, # Mixed Precision FP16
    resume_training=True
):
    print("=======================================================")
    print("STARTING OPTIMIZED OVERNIGHT TRAINING (2x T4 GPU + AMP FP16)")
    print(f"Data Dir: {data_dir} | Output Dir: {output_dir}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Save Step Freq: {save_step_frequency}")
    print("=======================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"[Trainer] Detected {num_gpus} GPU(s). Primary device: {device}")

    # 1. Load Datasets
    loader = MedPRSDatasetLoader(data_dir=data_dir)
    journal_df = loader.load_journals()
    num_classes = len(journal_df)
    print(f"[Trainer] Target journal classes: {num_classes}")

    train_df = loader.load_papers(split="train")
    val_df = loader.load_papers(split="val")

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    train_dataset = MedPRSTrainDataset(train_df, tokenizer)
    val_dataset = MedPRSTrainDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if num_gpus > 1 else 2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if num_gpus > 1 else 2
    )

    # 2. Initialize Model & Multi-GPU Wrapper
    base_model = BioBERTSimCPSREncoder().to(device)
    classifier_head = nn.Linear(768, num_classes).to(device)

    # ⚡ OPTIMIZATION 1: Multi-GPU DataParallel for 2x T4
    if num_gpus > 1:
        print(f"[Trainer] ⚡ Enabling Multi-GPU DataParallel across {num_gpus} GPUs!")
        model = nn.DataParallel(base_model)
        classifier = nn.DataParallel(classifier_head)
    else:
        model = base_model
        classifier = classifier_head

    # ⚡ OPTIMIZATION 2: Mixed Precision Scaler (FP16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)

    # ⚡ OPTIMIZATION 3: Resume Training Checkpoint (Auto-detecting existing Epoch3 checkpoint)
    latest_ckpt_path = os.path.join(output_dir, "latest_step_checkpoint.pth")
    if not os.path.exists(latest_ckpt_path):
        # Search input data_dir or /kaggle/input for existing Epoch3 checkpoint
        for s_dir in [data_dir, "/kaggle/input/"]:
            if s_dir and os.path.exists(s_dir):
                for root, dirs, files in os.walk(s_dir):
                    if root.endswith("Epoch3/latest_step_checkpoint") or "latest_step_checkpoint" in dirs:
                        latest_ckpt_path = os.path.join(root, "latest_step_checkpoint") if "latest_step_checkpoint" in dirs else root
                        break
                    for f in files:
                        if f.endswith(".pth") or f.endswith(".pt") or "simcprs" in f.lower():
                            latest_ckpt_path = os.path.join(root, f)
                            break

    if resume_training and os.path.exists(latest_ckpt_path):
        print(f"[Trainer] 🔄 Resuming training from checkpoint: {latest_ckpt_path}")
        try:
            # Handle unpacked directory format if needed
            w_path = latest_ckpt_path
            if os.path.isdir(latest_ckpt_path):
                data_pkl = os.path.join(latest_ckpt_path, "data.pkl")
                if not os.path.exists(data_pkl):
                    for root, dirs, files in os.walk(latest_ckpt_path):
                        if "data.pkl" in files:
                            w_path = root
                            break
                if os.path.exists(os.path.join(w_path, "data.pkl")):
                    import zipfile, tempfile
                    print(f"[Trainer] Re-packing PyTorch directory checkpoint '{w_path}' for resume...")
                    temp_zip_path = os.path.join(tempfile.gettempdir(), "temp_resume_ckpt.pt")
                    archive_name = os.path.basename(w_path.rstrip("/\\")) or "archive"
                    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_STORED) as zip_f:
                        for root, dirs, files in os.walk(w_path):
                            for file in files:
                                abs_path = os.path.join(root, file)
                                rel_path = os.path.relpath(abs_path, w_path)
                                zip_f.write(abs_path, os.path.join(archive_name, rel_path))
                    w_path = temp_zip_path

            try:
                ckpt = torch.load(w_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(w_path, map_location=device)

            if isinstance(w_path, str) and w_path.endswith("temp_resume_ckpt.pt") and os.path.exists(w_path):
                try: os.remove(w_path)
                except: pass

            saved_epoch = ckpt.get("epoch", 3)
            start_epoch = saved_epoch + 1 if saved_epoch < epochs else saved_epoch
            global_step = ckpt.get("global_step", 0)
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            
            raw_m = model.module if hasattr(model, 'module') else model
            raw_c = classifier.module if hasattr(classifier, 'module') else classifier
            
            if "model_state_dict" in ckpt:
                raw_m.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "classifier_head" in ckpt:
                raw_c.load_state_dict(ckpt["classifier_head"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except: pass
            if "scheduler_state_dict" in ckpt:
                try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except: pass

            print(f"[Trainer] 🚀 RESUME SUCCESSFUL! Loaded Epoch {saved_epoch} (Step {global_step}). Starting Epoch {start_epoch} -> Epoch {epochs}!")
        except Exception as e:
            print(f"[Trainer] Warning resuming checkpoint: {e}. Starting fresh.")

    # Helper function to save checkpoint ("Train tới đâu lưu tới đó")
    def save_checkpoint(save_name, epoch_num, step_num, current_val_acc):
        ckpt_path = os.path.join(output_dir, save_name)
        raw_m = model.module if hasattr(model, 'module') else model
        raw_c = classifier.module if hasattr(classifier, 'module') else classifier
        
        save_dict = {
            "epoch": epoch_num,
            "global_step": step_num,
            "model_state_dict": raw_m.state_dict(),
            "classifier_head": raw_c.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": current_val_acc,
            "timestamp": time.time()
        }
        torch.save(save_dict, ckpt_path)
        print(f"[Trainer] 💾 Saved Checkpoint to '{save_name}' (Step {step_num}, Epoch {epoch_num})")

    # --- Training Loop ---
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.train()
        classifier.train()

        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            global_step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # ⚡ Mixed Precision Autocast (FP16)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                embeddings = model(input_ids, attention_mask)
                logits = classifier(embeddings)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            # Progress logging every 200 steps
            if step % 200 == 0 or step == len(train_loader):
                elapsed = time.time() - start_time
                steps_per_sec = step / max(1.0, elapsed)
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} (Global {global_step}) | Loss: {total_loss / step:.4f} | Speed: {steps_per_sec:.1f} steps/s | Elapsed: {elapsed/60:.1f}m")

            # ⚡ STEP CHECKPOINT: "Train tới đâu lưu tới đó" (Lưu mỗi save_step_frequency steps)
            if global_step % save_step_frequency == 0:
                save_checkpoint("latest_step_checkpoint.pth", epoch, global_step, best_val_acc)

        # End of Epoch Checkpoint
        save_checkpoint(f"epoch_{epoch}_checkpoint.pth", epoch, global_step, best_val_acc)

        # Validation Phase
        print(f"\n[Validation] Evaluating Epoch {epoch} on Validation Set...")
        model.eval()
        classifier.eval()

        val_hits_1, val_hits_10, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    embeddings = model(input_ids, attention_mask)
                    logits = classifier(embeddings)

                _, top1_preds = torch.max(logits, dim=1)
                val_hits_1 += (top1_preds == labels).sum().item()

                _, top10_preds = torch.topk(logits, k=10, dim=1)
                val_hits_10 += (top10_preds == labels.unsqueeze(1)).sum().item()

                val_total += labels.size(0)

        val_acc1 = (val_hits_1 / val_total) * 100.0
        val_acc10 = (val_hits_10 / val_total) * 100.0
        print(f"==> Epoch {epoch} Validation Top-1 Acc: {val_acc1:.2f}% | Top-10 Acc: {val_acc10:.2f}%")

        if val_acc10 > best_val_acc:
            best_val_acc = val_acc10
            save_checkpoint("best_simcprs_checkpoint.pth", epoch, global_step, best_val_acc)
            print(f"🎉 NEW BEST MODEL! (Val Top-10: {val_acc10:.2f}%) saved to 'best_simcprs_checkpoint.pth'")

    print("\n=======================================================")
    print(f"OVERNIGHT TRAINING COMPLETED! Best Val Top-10 Acc: {best_val_acc:.2f}%")
    print(f"Saved Checkpoints in: {output_dir}")
    print("=======================================================")

if __name__ == "__main__":
    train_overnight()
