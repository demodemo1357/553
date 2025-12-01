import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def plot_kmeans_clusters(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_tfidf = vectorizer.fit_transform(df['text'])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X_tfidf)
    X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())
    df['x'] = X_pca[:, 0]
    df['y'] = X_pca[:, 1]
    print("\n=== Cluster/Source Counts ===")
    print(df.groupby(['cluster', 'source']).size())
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x='x', y='y',
        hue='cluster', palette='Set2', s=50
    )
    plt.title("KMeans Results", fontsize=16)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()


def plot_source_scatter(df):
    plt.figure(figsize=(10, 6))

    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='source',
        palette=custom_colors,
        s=50
    )

    plt.title("Real Results", fontsize=16)
    plt.xlabel("PCA 1", fontsize=14)
    plt.ylabel("PCA 2", fontsize=14)
    plt.legend(title='Source')
    plt.tight_layout()
    plt.show()

def plot_length_histogram(df_true, df_fake, df_neural):
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))

    plt.hist(df_true['length'], bins=40, alpha=0.6, label='True')
    plt.hist(df_fake['length'], bins=40, alpha=0.6, label='Fake')
    plt.hist(df_neural['length'], bins=40, alpha=0.6, label='Neural')

    plt.legend(frameon=True)
    plt.title("Text Length Distribution", fontsize=16)
    plt.xlabel("Text Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_classification_report_bars(precision, recall, f1, labels, title="Model Performance"):
    x = np.arange(len(labels))
    width = 0.18
    gap = 0.08  # separation inside each group

    plt.figure(figsize=(10, 6))

    plt.bar(x - width - gap, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width + gap, f1, width, label='F1 Score')

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df_all = pd.read_csv("benchmark.csv")
    df_all["length"] = df_all["text"].str.len()
    df_true = df_all[df_all['source'] == 0]
    df_fake = df_all[df_all['source'] == 1]
    df_neural = df_all[df_all['source'] == 2]
    plot_kmeans_clusters(df_all)
    plot_source_scatter(df_all)
    plot_length_histogram(df_true, df_fake, df_neural)
    precision = [0.947, 0.826, 0.914]
    recall =    [0.938, 0.714, 0.944]
    f1 =        [0.943, 0.766, 0.928]
    labels = ["human_fake", "AI", "true"]
    plot_classification_report_bars(
        precision, recall, f1, labels,
        title="Logistic Regression Results"
    )

if __name__ == "__main__":
    main()
