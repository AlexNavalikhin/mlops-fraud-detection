import pandas as pd
import json
import os
from datetime import datetime

class RawStorage:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)

    def save_batch(self, df, meta, batch_index):
        batch_dir = os.path.join(self.raw_dir, f"batch_{batch_index:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        df.to_csv(os.path.join(batch_dir, "data.csv"), index=False)
        meta['saved_at'] = datetime.now().isoformat()
        with open(os.path.join(batch_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def get_next_batch_index(self):
        existing = [
            d for d in os.listdir(self.raw_dir)
            if d.startswith("batch_")
        ]
        return len(existing)

    def load_all_batches(self):
        frames = []
        for d in sorted(os.listdir(self.raw_dir)):
            path = os.path.join(self.raw_dir, d, "data.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_meta_history(self):
        history = []
        for d in sorted(os.listdir(self.raw_dir)):
            path = os.path.join(self.raw_dir, d, "meta.json")
            if os.path.exists(path):
                with open(path) as f:
                    history.append(json.load(f))
        return history
