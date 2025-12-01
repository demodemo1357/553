import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer


class BenchmarkClassifier:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.label_encoder = LabelEncoder()

    def load_data(self, file_path):
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        df["text"] = df["text"].astype(str)
        return df

    def embed(self, texts):
        print(f"Encoding {len(texts)} texts into embeddings...")
        return self.embedder.encode(texts, show_progress_bar=True)

    def train_and_eval(self, X_train, X_test, y_train, y_test, model_type):
        if model_type.lower() == "svm":
            print("Training Linear SVM...")
            clf = LinearSVC(random_state=42)
        else:
            print("Training Logistic Regression...")
            clf = LogisticRegression(max_iter=1000, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("\nEvaluation Results")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n",
              classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, digits=3))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def run(self, csv_file, model_choice):
        df = self.load_data(csv_file)

        texts = df["text"].tolist()
        labels = self.label_encoder.fit_transform(df["source"])

        embeddings = self.embed(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.train_and_eval(X_train, X_test, y_train, y_test, model_choice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Text Classifier")
    parser.add_argument("--file", type=str, required=True, help="Path to benchmark CSV file")
    parser.add_argument("--model", type=str, choices=["svm", "lr"], default="svm",
                        help="Choose classifier: svm or lr")

    args = parser.parse_args()

    classifier = BenchmarkClassifier()
    classifier.run(args.file, args.model)