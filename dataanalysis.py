import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Database setup
DATABASE_URL = "sqlite:///reddit_data.db"
engine = create_engine(DATABASE_URL)

# Load data from SQLite
def load_data():
    query = "SELECT * FROM reddit_posts"
    df = pd.read_sql(query, engine, parse_dates=['created_utc'])
    return df

# Plot distribution of post scores
def plot_score_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['score'], bins=20, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Post Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('score_distribution.png')
    plt.show()

# Plot box plot of post scores
def plot_boxplot(df):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['score'], vert=False)
    plt.title('Box Plot of Post Scores')
    plt.xlabel('Score')
    plt.grid(True)
    plt.savefig('box_plot.png')
    plt.show()

# Plot heatmap of average scores by day and month
def plot_heatmap(df):
    df['day'] = df.index.day
    df['month'] = df.index.month
    pivot_table = df.pivot_table(index='day', columns='month', values='score', aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.1f')
    plt.title('Heatmap of Average Scores by Day and Month')
    plt.xlabel('Month')
    plt.ylabel('Day')
    plt.savefig('heatmap_scores.png')
    plt.show()

def main():
    df = load_data()
    df.set_index('created_utc', inplace=True)

    plot_score_distribution(df)
    plot_boxplot(df)
    plot_heatmap(df)

if __name__ == "__main__":
    main()
