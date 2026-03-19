"""
fastapi/scripts/train.py
-------------------------
BERT fine-tuning pipeline for support ticket classification.
All paths are relative to the fastapi/ folder (the project root for this env).

Usage (run from fastapi/):
    python scripts/train.py

Optional args:
    --data_dir    path to data folder        (default: data/)
    --model_dir   path to save checkpoints   (default: models/)
    --mlflow_dir  MLflow tracking store      (default: mlflow_runs/)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
import os 

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 4
LR          = 2e-5
WARMUP_RATIO = 0.1
CATEGORIES  = ["Delivery", "Refund", "Account", "Product Issue", "Other"]

torch.manual_seed(SEED)
np.random.seed(SEED)


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(self.labels[idx], dtype=torch.long)}


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out  = model(input_ids=batch["input_ids"].to(device),
                     attention_mask=batch["attention_mask"].to(device),
                     labels=batch["labels"].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            total_loss += out.loss.item()
            all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def main(args):
    # Anchor all paths to fastapi/ (parent of scripts/) regardless of cwd
    _script_dir = Path(__file__).resolve().parent.parent  # scripts/ -> fastapi/
    data_dir    = (_script_dir / args.data_dir).resolve()
    model_dir   = (_script_dir / args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard: timestamped log dir under logs/bert/ ─────────────────
    # Placed beside logs/fit/ (the CNN log dir) so a single TB instance
    # launched with --logdir logs/ shows both models side by side.
    base_dir   = _script_dir
    ts         = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = str(base_dir / "logs" / "bert" / ts)
    writer     = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log dir: {tb_log_dir}")

    # Load & merge
    df = pd.read_csv(data_dir / "Support_Tickets_Unlabeled.csv").merge(
         pd.read_csv(data_dir / "ticket_labels.csv"), on="ticket_id")
    print(f"Loaded {len(df)} labeled tickets\n{df['category'].value_counts()}\n")

    le = LabelEncoder()
    le.fit(CATEGORIES)
    df["label_id"] = le.transform(df["category"])

    label_map = {int(le.transform([c])[0]): c for c in CATEGORIES}
    with open(model_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label_id"])
    val_df,  test_df  = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"])
    print(f"Split — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    for split, dframe in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dframe.to_csv(data_dir / f"{split}.csv", index=False)

    tokenizer    = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(TicketDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TicketDataset(val_df["text"].tolist(),   val_df["label_id"].tolist(),   tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TicketDataset(test_df["text"].tolist(),  test_df["label_id"].tolist(),  tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CATEGORIES)).to(device)

    # ── Write model graph to TensorBoard ──────────────────────────────────
    # Feed a dummy batch of token ids so TB can trace the forward pass.
    # Note: BERT graphs are large (~400 nodes); most useful for confirming
    # the architecture and inspecting the classification head.
    try:
        dummy_ids  = torch.zeros(1, MAX_LEN, dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long).to(device)
        # strict=False required — BERT returns a dict (BaseModelOutput) from its
        # forward pass, which torch.jit.trace cannot handle in strict mode
        writer.add_graph(model, (dummy_ids, dummy_mask), use_strict_trace=False)
        print("Model graph written to TensorBoard.")
    except Exception as e:
        print(f"Warning: could not write model graph to TensorBoard: {e}")

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    # ── Add hyperparameters to TensorBoard HParams tab ────────────────────
    from torch.utils.tensorboard.summary import hparams as tb_hparams
    hparam_dict   = {"model": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
                     "epochs": EPOCHS, "lr": LR, "warmup_ratio": WARMUP_RATIO, "seed": SEED}
    metric_dict   = {"hparam/val_f1": 0.0, "hparam/test_acc": 0.0, "hparam/test_f1": 0.0}

    # Resolve mlflow_dir relative to fastapi/ — already have _script_dir from above
    #_mlflow_path = (_script_dir / args.mlflow_dir).resolve()
    # file:/// prefix required on Windows — without it MLflow tries to parse
    # the drive letter (C:) as a URI scheme
    #mlflow.set_tracking_uri("file:///" + str(_mlflow_path).replace("\\", "/"))

    # Using mlflow:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("support_ticket_classification")

    # --- Per-epoch history (saved to JSON for Streamlit) --------------------------------
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    with mlflow.start_run(run_name="bert_base_uncased") as run:
        mlflow.log_params({"model": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
                           "epochs": EPOCHS, "lr": LR, "seed": SEED,
                           "train": len(train_df), "val": len(val_df), "test": len(test_df)})
        best_val_f1 = 0.0

        for epoch in range(1, EPOCHS + 1):
            tl = train_epoch(model, train_loader, optimizer, scheduler, device)
            vl, va, vf, _, _ = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}/{EPOCHS}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}  val_f1={vf:.4f}")

            # MLflow
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl,
                                 "val_accuracy": va, "val_f1_weighted": vf}, step=epoch)

            # TensorBoard scalars
            writer.add_scalars("Loss",     {"train": tl, "val": vl}, epoch)
            writer.add_scalars("Accuracy", {"val":   va},            epoch)
            writer.add_scalars("F1",       {"val":   vf},            epoch)

            # Per-epoch history for Streamlit JSON
            history["train_loss"].append(round(tl, 6))
            history["val_loss"].append(round(vl, 6))
            history["val_accuracy"].append(round(va, 6))
            history["val_f1"].append(round(vf, 6))

            if vf > best_val_f1:
                best_val_f1 = vf
                model.save_pretrained(model_dir / "best_checkpoint")
                tokenizer.save_pretrained(model_dir / "best_checkpoint")
                print(f"Saved best model (val_f1={vf:.4f})")

        # -- Test evaluation on best checkpoint --------------------------------------------------------
        best_model = BertForSequenceClassification.from_pretrained(
            model_dir / "best_checkpoint").to(device)
        _, test_acc, test_f1, test_preds, test_labels_list = evaluate(
            best_model, test_loader, device)
        print(f"\nTest — accuracy: {test_acc:.4f}  f1_weighted: {test_f1:.4f}")
        print(classification_report(test_labels_list, test_preds, target_names=le.classes_))

        report = classification_report(
            test_labels_list, test_preds, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(test_labels_list, test_preds)

        pd.DataFrame(report).T.to_csv(model_dir / "classification_report.csv")
        pd.DataFrame(cm, index=le.classes_,
                     columns=le.classes_).to_csv(model_dir / "confusion_matrix.csv")

        # -- Save predicted probabilities for confidence density plot --------------------------------
        # Re-run test set through best model to collect softmax probs
        best_model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                logits = best_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                ).logits
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pred_probs = np.vstack(all_probs)
        np.save(model_dir / "test_pred_probs.npy",  pred_probs)
        np.save(model_dir / "test_y_true.npy",      np.array(test_labels_list))
        mlflow.log_artifact(str(model_dir / "test_pred_probs.npy"))  # ← must be inside with block
        mlflow.log_artifact(str(model_dir / "test_y_true.npy"))      # ← must be inside with block
        print("Saved and logged test_pred_probs.npy and test_y_true.npy")

        # -- Save per-epoch history JSON ------------------------------------------
        with open(model_dir / "bert_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print("Saved bert_history.json")

        # -- Save model card JSON ---------------------------------------------------
        model_card = {
            # Identity
            "model":          MODEL_NAME,
            "architecture":   "BertForSequenceClassification",
            "framework":      "PyTorch/HuggingFace Transformers",
            "trained_on":     str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),

            # MLflow reference — replaces local checkpoint_dir
            "mlflow_run_id":  run.info.run_id,
            "mlflow_model":   "bert-ticket-classifier",
            "mlflow_stage":   "Production",

            # TensorBoard
            "tensorboard_logdir": str(Path(tb_log_dir).relative_to(base_dir)).replace("\\", "/"),

            # Hyperparameters
            "hyperparameters": {
                "max_len":      MAX_LEN,
                "batch_size":   BATCH_SIZE,
                "epochs":       EPOCHS,
                "lr":           LR,
                "warmup_ratio": WARMUP_RATIO,
                "seed":         SEED,
            },

            # Data
            "data_split": {
                "train": len(train_df),
                "val":   len(val_df),
                "test":  len(test_df),
            },

            # Metrics
            "metrics": {
                "best_val_f1":   round(best_val_f1, 4),
                "test_accuracy": round(test_acc, 4),
                "test_f1":       round(test_f1, 4),
                "per_class_f1": {
                    c: round(report[c]["f1-score"], 4) for c in CATEGORIES
                },
            },

            # Intended use
            "intended_use":  "Support ticket classification into 5 categories",
            "input":         "Free text support ticket (max 128 tokens)",
            "output":        "Category + confidence score + per-class scores",
            "limitations":   "Trained on 400 tickets — may struggle with ambiguous tickets",
            "categories":    CATEGORIES,
        }
        with open(model_dir / "bert_model_card.json", "w") as f:
            json.dump(model_card, f, indent=2)
        print("Saved bert_model_card.json")

        # ── TensorBoard: final test metrics + HParams ─────────────────────
        writer.add_scalar("Test/accuracy",  test_acc, EPOCHS)
        writer.add_scalar("Test/f1",        test_f1,  EPOCHS)
        # add_hparams requires tensorboard's hparams plugin which uses
        # np.string_ — removed in NumPy 2.0. Wrapped so training still
        # completes; upgrade tensorboard to fix: uv add tensorboard --upgrade
        try:
            writer.add_hparams(
                hparam_dict,
                {"hparam/val_f1":   best_val_f1,
                 "hparam/test_acc": test_acc,
                 "hparam/test_f1":  test_f1},
            )
        except Exception as _hp_err:
            print(f"Warning: could not write HParams to TensorBoard: {_hp_err}")
            print("  To fix: uv add tensorboard --upgrade")
        writer.flush()
        writer.close()

        # -- MLflow artefacts ----------------------------------------------------------------
        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_weighted": test_f1,
                             **{f"test_f1_{c.replace(' ','_').lower()}": report[c]["f1-score"]
                                for c in CATEGORIES}})
        for fname in ["classification_report.csv", "confusion_matrix.csv",
                      "label_map.json", "bert_history.json", "bert_model_card.json"]:
            mlflow.log_artifact(str(model_dir / fname))
      
        # Logging as HuggingFace transformers model
        mlflow.transformers.log_model(
            transformers_model={"model": best_model, "tokenizer": tokenizer},
            name="bert_classifier",
            registered_model_name="bert-ticket-classifier",
            task="text-classification",
            pip_requirements=[
                "transformers==" + __import__("transformers").__version__,
                "torch==" + __import__("torch").__version__,
                "tokenizers==" + __import__("tokenizers").__version__,
            ]
        )

    print("\nTraining complete.")
    print(f"  Artefacts  -> {model_dir}")
    print(f"  TensorBoard -> {tb_log_dir}")
    print(f"  MLflow run  -> {run.info.run_id}")

    # ── Promote to Production ─────────────────────────────────────────────
    client   = MlflowClient()
    versions = client.get_latest_versions("bert-ticket-classifier", stages=["None"])
    if versions:
        client.transition_model_version_stage(
            name="bert-ticket-classifier",
            version=versions[0].version,
            stage="Production"
        )
        print(f"BERT model v{versions[0].version} promoted to Production")
    else:
        print("[WARN] No new versions found to promote — already in Production?")

    # ---Verify all artifacts uploaded correctly -----------------------------------
    print("\nVerifying MLflow artifacts...")
    versions = client.get_latest_versions("bert-ticket-classifier", stages=["None"])
    if not versions:
        versions = client.get_latest_versions("bert-ticket-classifier", stages=["Production"])
    if versions:
        run_id    = versions[0].run_id
        artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
        uploaded  = {a.path for a in artifacts}

        required = [
            "bert_model_card.json",
            "bert_history.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "label_map.json",
            "test_pred_probs.npy",
            "test_y_true.npy",
        ]

        all_ok = True
        for f in required:
            if f in uploaded:
                print(f"  [OK]      {f}")
            else:
                print(f"  [MISSING] {f} ← not uploaded!")
                all_ok = False

        if all_ok:
            print("\nAll artifacts verified ✅")
        else:
            print("\n[WARN] Some artifacts missing — check logs above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--model_dir",  default="models/")
    # mlflow_dir removed — MLflow URI now set via MLFLOW_TRACKING_URI env var
    # Local default: http://localhost:5000
    # Docker:        http://mlflow:5000 (set in docker-compose.yml)
    main(parser.parse_args())
