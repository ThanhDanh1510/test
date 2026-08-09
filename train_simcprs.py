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
    epochs=10,
    batch_size=32,  # Tối ưu 32 (16 per GPU) tránh tràn VRAM GPU
    lr=5e-5,
    save_step_frequency=1000,
    use_fp16=True,
    resume_training=True,
    epoch2_checkpoint="/kaggle/input/datasets/tintngc/medprs-dataset/Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth"
):
    print("=======================================================")
    print("STARTING CONTINUOUS SIMCPSR TRAINING (EPOCH 3 -> EPOCH 10)")
    print(f"Data Dir: {data_dir} | Output Dir: {output_dir}")
    print(f"Target Epochs: {epochs} | Batch Size: {batch_size} | Save Step Freq: {save_step_frequency}")
    print("=======================================================\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"[Trainer] Detected {num_gpus} GPU(s). Primary device: {device}")

    # 1. Load Datasets & Journals
    loader = MedPRSDatasetLoader(data_dir=data_dir)
    journal_df = loader.load_journals()
    num_classes = len(journal_df)
    print(f"[Trainer] Loaded {num_classes} journal classes.")

    train_df = loader.load_papers(split="train")
    val_df = loader.load_papers(split="val")

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    train_dataset = MedPRSTrainDataset(train_df, tokenizer)
    val_dataset = MedPRSTrainDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # 2. Pre-tokenize all journal titles for journal_proj calculation
    print("[Trainer] Pre-tokenizing 1,406 journal titles for dual-branch training...")
    j_titles = [str(r['title']).strip() for _, r in journal_df.iterrows()]
    j_tokenized = tokenizer(j_titles, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    j_input_ids = j_tokenized['input_ids'].to(device)
    j_attn_mask = j_tokenized['attention_mask'].to(device)

    # Helper function to compute journal projected tensor in mini-batches safely
    def compute_journal_proj_tensor(model_obj):
        m_eval = model_obj.module if hasattr(model_obj, 'module') else model_obj
        m_eval.eval()
        j_embeddings = []
        with torch.no_grad():
            for i in range(0, len(j_input_ids), 128):
                b_ids = j_input_ids[i:i+128]
                b_mask = j_attn_mask[i:i+128]
                try:
                    autocast_ctx = torch.amp.autocast('cuda', enabled=use_fp16)
                except AttributeError:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16)
                with autocast_ctx:
                    emb = m_eval.encode_journal(b_ids, b_mask)
                j_embeddings.append(emb)
        j_proj = torch.cat(j_embeddings, dim=0).detach()
        m_eval.train()
        return j_proj

    # 3. Initialize SimCPSR Model Architecture
    raw_model = BioBERTSimCPSREncoder(num_classes=num_classes).to(device)

    start_epoch = 3  # Resuming from Epoch 2 completion
    global_step = 0
    best_val_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)

    # 4. Auto-detect & Load Epoch 2 Checkpoint
    target_ckpt = None
    check_candidates = [
        os.path.join(output_dir, "latest_step_checkpoint.pth"),
        epoch2_checkpoint,
        os.path.join(data_dir, "Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth")
    ]
    for c in check_candidates:
        if c and os.path.exists(c):
            target_ckpt = c
            break

    if not target_ckpt and os.path.exists("/kaggle/input/"):
        for root, dirs, files in os.walk("/kaggle/input/"):
            for f in files:
                if "epoch_02" in f.lower() or "epoch02" in f.lower() or "latest_step" in f.lower():
                    target_ckpt = os.path.join(root, f)
                    break
            if target_ckpt: break

    if resume_training and target_ckpt and os.path.exists(target_ckpt):
        print(f"[Trainer] 🔄 Resuming training from checkpoint: {target_ckpt}")
        try:
            w_path = target_ckpt
            if os.path.isdir(target_ckpt):
                data_pkl = os.path.join(target_ckpt, "data.pkl")
                if not os.path.exists(data_pkl):
                    for root, dirs, files in os.walk(target_ckpt):
                        if "data.pkl" in files:
                            w_path = root
                            break
                if os.path.exists(os.path.join(w_path, "data.pkl")):
                    import zipfile, tempfile
                    print(f"[Trainer] Re-packing directory checkpoint '{w_path}'...")
                    temp_zip = os.path.join(tempfile.gettempdir(), "temp_train_ckpt.pt")
                    archive_name = os.path.basename(w_path.rstrip("/\\")) or "archive"
                    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_STORED) as zf:
                        for root, dirs, files in os.walk(w_path):
                            for file in files:
                                abs_p = os.path.join(root, file)
                                rel_p = os.path.relpath(abs_p, w_path)
                                zf.write(abs_p, os.path.join(archive_name, rel_p))
                    w_path = temp_zip

            try:
                ckpt = torch.load(w_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(w_path, map_location=device)

            if isinstance(w_path, str) and w_path.endswith("temp_train_ckpt.pt") and os.path.exists(w_path):
                try: os.remove(w_path)
                except: pass

            state_dict = ckpt
            if isinstance(ckpt, dict):
                if "epoch" in ckpt:
                    saved_ep = ckpt["epoch"]
                    start_epoch = saved_ep + 1 if saved_ep < epochs else saved_ep
                    global_step = ckpt.get("global_step", 0)
                    best_val_acc = ckpt.get("best_val_acc", 0.0)

                if "model_state_dict" in ckpt:
                    state_dict = dict(ckpt["model_state_dict"])
                    if "classifier_head" in ckpt and isinstance(ckpt["classifier_head"], dict):
                        for k, v in ckpt["classifier_head"].items():
                            state_dict[k] = v
                elif "state_dict" in ckpt:
                    state_dict = dict(ckpt["state_dict"])

            model_dict = raw_model.state_dict()
            cleaned_state = {}
            matched = 0

            for k, v in state_dict.items():
                clean_k = k
                for prefix in ["base_model.bert.", "base_model.", "paper_encoder.", "encoder.", "module."]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                        break

                if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                    cleaned_state[clean_k] = v
                    matched += 1
                elif f"encoder.{clean_k}" in model_dict and model_dict[f"encoder.{clean_k}"].shape == v.shape:
                    cleaned_state[f"encoder.{clean_k}"] = v
                    matched += 1
                elif k in model_dict and model_dict[k].shape == v.shape:
                    cleaned_state[k] = v
                    matched += 1

            raw_model.load_state_dict(cleaned_state, strict=False)
            print(f"[Trainer] 🚀 SUCCESSFUL RESUME! Loaded {matched}/{len(model_dict)} layers from {target_ckpt}. Starting Epoch {start_epoch} -> Epoch {epochs}!")
        except Exception as e:
            print(f"[Trainer] Warning resuming checkpoint: {e}. Starting fresh at Epoch 3.")

    # Multi-GPU DataParallel setup
    if num_gpus > 1:
        print(f"[Trainer] ⚡ Enabling Multi-GPU DataParallel across {num_gpus} GPUs!")
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * (epochs - start_epoch + 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    criterion = nn.CrossEntropyLoss()

    def save_checkpoint(save_name, epoch_num, step_num, current_val_acc):
        ckpt_path = os.path.join(output_dir, save_name)
        m_save = model.module if hasattr(model, 'module') else model
        save_dict = {
            "epoch": epoch_num,
            "global_step": step_num,
            "model_state_dict": m_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": current_val_acc,
            "timestamp": time.time()
        }
        torch.save(save_dict, ckpt_path)
        print(f"[Trainer] 💾 Saved Checkpoint to '{save_name}' (Step {step_num}, Epoch {epoch_num})")

    # --- Training Loop ---
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n==========================================")
        print(f"STARTING EPOCH {epoch}/{epochs}")
        print(f"==========================================")
        
        # Pre-compute journal embeddings safely with detach()
        j_proj_tensor = compute_journal_proj_tensor(model)

        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            global_step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            try:
                autocast_ctx = torch.amp.autocast('cuda', enabled=use_fp16)
            except AttributeError:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16)

            with autocast_ctx:
                m_curr = model.module if hasattr(model, 'module') else model
                paper_proj = m_curr.encode_paper(input_ids, attention_mask)
                logits = m_curr.forward_logits(paper_proj, j_proj_tensor)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                elapsed = time.time() - start_time
                steps_per_sec = step / max(1.0, elapsed)
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} (Global {global_step}) | Loss: {total_loss / step:.4f} | Speed: {steps_per_sec:.1f} steps/s | Elapsed: {elapsed/60:.1f}m")

            if global_step % save_step_frequency == 0:
                save_checkpoint("latest_step_checkpoint.pth", epoch, global_step, best_val_acc)

        save_checkpoint(f"epoch_{epoch}_checkpoint.pth", epoch, global_step, best_val_acc)

        # Validation Loop
        print(f"\n[Validation] Evaluating Epoch {epoch} on Validation Set...")
        val_hits_1, val_hits_10, val_total = 0, 0, 0
        j_proj_val = compute_journal_proj_tensor(model)
        m_eval = model.module if hasattr(model, 'module') else model
        m_eval.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    paper_proj = m_eval.encode_paper(input_ids, attention_mask)
                    logits = m_eval.forward_logits(paper_proj, j_proj_val)

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
    print(f"TRAINING COMPLETED! Best Val Top-10 Acc: {best_val_acc:.2f}%")
    print(f"Saved Checkpoints in: {output_dir}")
    print("=======================================================")

if __name__ == "__main__":
    train_overnight()
