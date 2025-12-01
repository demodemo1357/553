import pandas as pd
import re
from datasets import load_dataset
from sklearn.utils import resample
import argparse
LABEL_MAP = {
    "neural": 1,
    "real": 0
}
class BenchmarkPreprocessor:
    def __init__(self, max_length=5000, min_length=750, random_state=42):
        self.max_length = max_length
        self.min_length = min_length
        self.random_state = random_state
    @staticmethod
    def clean_text(text):
        if text is None:
            return ""
        text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    @staticmethod
    def delete_prefix(text):
        if pd.isna(text):
            return ""
        if '-' in text:
            text = text.split('-', 1)[1]
        return text.strip()
    @staticmethod
    def process_dataset_split(ds_split):
        """Process HuggingFace dataset split into list of dicts."""
        processed = [
            {
                "label": LABEL_MAP[item["label"]],
                "text": BenchmarkPreprocessor.clean_text(item["body"])
            }
            for item in ds_split if item["body"] is not None and item["language"] == "en"
        ]
        return processed
    def load_neural_news(self):
        """Load and preprocess neural news benchmark dataset."""
        ds = load_dataset("tum-nlp/neural-news-benchmark")
        train_data = self.process_dataset_split(ds['train'])
        valid_data = self.process_dataset_split(ds['validation'])
        test_data = self.process_dataset_split(ds['test'])
        all_data = train_data + valid_data + test_data
        df = pd.DataFrame(all_data)
        return df
    def preprocess_csv(self, df_fake_path=None, df_true_path=None):
        """Load CSV files and preprocess them."""
        df_fake = pd.read_csv(df_fake_path) if df_fake_path else pd.DataFrame(columns=['text'])
        df_true = pd.read_csv(df_true_path) if df_true_path else pd.DataFrame(columns=['text'])
        df_true['text'] = df_true['text'].apply(self.delete_prefix).apply(self.clean_text)
        df_fake['text'] = df_fake['text'].apply(self.clean_text)
        df_neural = self.load_neural_news()
        df_neural = df_neural[df_neural['label'] == 1].drop(columns=['label']).drop_duplicates(subset='text').reset_index(drop=True)
        df_real = df_neural[df_neural['label'] == 0].drop(columns=['label']).drop_duplicates(subset='text').reset_index(drop=True)
        df_true = pd.concat([df_true, df_real], ignore_index=True).drop_duplicates(subset='text').reset_index(drop=True)
        df_fake = df_fake.drop_duplicates(subset='text').reset_index(drop=True)
        for df_ in [df_true, df_fake, df_neural]:
            df_['length'] = df_['text'].str.len()
            df_ = df_[(df_['length'] <= self.max_length) & (df_['length'] >= self.min_length)].reset_index(drop=True)
        df_true['source'] = 'true'
        df_fake['source'] = 'fake'
        df_neural['source'] = 'neural'
        df_all = pd.concat([df_true, df_fake, df_neural], ignore_index=True)
        df_all['text'] = df_all['text'].astype(str)
        return df_all, df_true, df_fake, df_neural

    def balance_upsample(self, df_all):
        """Upsample neural class to match largest class (true)."""
        df_true = df_all[df_all['source'] == 'true']
        df_fake = df_all[df_all['source'] == 'fake']
        df_neural = df_all[df_all['source'] == 'neural']

        df_neural_upsampled = resample(
            df_neural,
            replace=True,
            n_samples=len(df_true),
            random_state=self.random_state
        )
        df_balanced = pd.concat([df_true, df_fake, df_neural_upsampled]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        return df_balanced

    def balance_downsample(self, df_all):
        """Downsample true and fake to match neural."""
        df_true = df_all[df_all['source'] == 'true']
        df_fake = df_all[df_all['source'] == 'fake']
        df_neural = df_all[df_all['source'] == 'neural']

        df_true_down = resample(df_true, replace=False, n_samples=len(df_neural), random_state=self.random_state)
        df_fake_down = resample(df_fake, replace=False, n_samples=len(df_neural), random_state=self.random_state)

        df_balanced = pd.concat([df_true_down, df_fake_down, df_neural]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        return df_balanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess benchmark CSV and neural news dataset.")
    parser.add_argument("--fake_csv", type=str, default="fake.csv", help="Path to fake CSV file")
    parser.add_argument("--true_csv", type=str, default="true.csv", help="Path to true CSV file")
    parser.add_argument("--output_csv", type=str, default="benchmark.csv", help="Output CSV path")
    args = parser.parse_args()
    preprocessor = BenchmarkPreprocessor()
    df_all, df_true, df_fake, df_neural = preprocessor.preprocess_csv(args.fake_csv, args.true_csv)
    df_balanced_upsample = preprocessor.balance_upsample(df_all)
    df_balanced_downsample = preprocessor.balance_downsample(df_all)
    df_all.to_csv(args.output_csv, index=False)
