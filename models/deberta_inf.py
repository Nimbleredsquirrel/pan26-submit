import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


class _TextDataset(Dataset):
    def __init__(self, recs, tok, max_len):
        self.recs = recs
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok(
            r["text"], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
        }


class DeBERTaInf:
    def __init__(self, ckpt_path, mdl_nm="microsoft/deberta-v3-large", max_len=256, batch_sz=16):
        self.max_len = max_len
        self.batch_sz = batch_sz

        if torch.cuda.is_available():
            self.dev = "cuda"
        elif torch.backends.mps.is_available():
            self.dev = "mps"
        else:
            self.dev = "cpu"

        print(f"loading DeBERTa from {ckpt_path} on {self.dev}")
        self.tok = AutoTokenizer.from_pretrained(mdl_nm)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            mdl_nm, num_labels=1, torch_dtype=torch.float32
        )
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.dev)
        self.model.eval()

    def predict_proba(self, records):
        ds = _TextDataset(records, self.tok, self.max_len)
        dl = DataLoader(ds, batch_size=self.batch_sz, shuffle=False)

        all_proba = []
        with torch.no_grad():
            for batch in dl:
                ids = batch["input_ids"].to(self.dev)
                msk = batch["attention_mask"].to(self.dev)
                logits = self.model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
                proba = torch.sigmoid(logits.float()).cpu().numpy()
                all_proba.extend(proba.tolist())

        return [r["id"] for r in records], all_proba
