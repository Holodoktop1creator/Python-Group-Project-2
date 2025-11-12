
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11 

COLORS = sns.color_palette("husl", 8)


def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} books\n") # сколько данных загрузилось 
    return df


def plot_distributions(df, save_path=None): # 4 графика на одном про распределение
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Distributions', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    sns.histplot(df['rating'], bins=50, kde=True, color=COLORS[0], ax=ax1)
    ax1.axvline(df['rating'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["rating"].mean():.2f}')
    ax1.axvline(df['rating'].median(), color='green', linestyle='--',
                label=f'Median: {df["rating"].median():.2f}')
    ax1.set_title('Rating Distribution (normal)')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2 = axes[0, 1]
    prices = df['price'].dropna()
    sns.histplot(prices, bins=50, kde=True, color=COLORS[1], ax=ax2)
    ax2.axvline(prices.mean(), color='red', linestyle='--',
                label=f'Mean: ${prices.mean():.2f}')
    ax2.axvline(prices.median(), color='green', linestyle='--',
                label=f'Median: ${prices.median():.2f}')
    ax2.set_title('Price Distribution')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    ax3 = axes[1, 0]
    sns.histplot(prices, bins=50, kde=False, color=COLORS[2], ax=ax3, log_scale=(True, False))
    ax3.set_title('Price Distribution (log scale)')
    ax3.set_xlabel('Price ($, log scale)')
    ax3.set_ylabel('Frequency')

    ax4 = axes[1, 1]
    sns.histplot(df['ratings_count'], bins=50, kde=False, color=COLORS[3],
                 ax=ax4, log_scale=(True, False))
    ax4.set_title('Ratings Count Distribution (log scale)')
    ax4.set_xlabel('Ratings Count (log scale)')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    

    plt.close('all')


def plot_correlations(df, save_path=None): # Про корреляции график 4 на одном 
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Correlations and Dependencies', fontsize=16, fontweight='bold')

    numeric_cols = ['price', 'rating', 'ratings_count']
    df_numeric = df[numeric_cols].dropna()

    ax1 = fig.add_subplot(gs[0, :])
    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Correlation Matrix')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df['ratings_count'], df['rating'],
                alpha=0.3, s=20, c=COLORS[4])
    ax2.set_xlabel('Ratings Count')
    ax2.set_ylabel('Rating')
    ax2.set_title('Regression to Mean: Rating vs Ratings Count')
    ax2.axhline(df['rating'].mean(), color='red', linestyle='--',
                label=f'Mean rating: {df["rating"].mean():.2f}')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    df_with_price = df.dropna(subset=['price'])
    ax3.scatter(df_with_price['price'], df_with_price['rating'],
                alpha=0.3, s=20, c=COLORS[5])
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Rating')
    ax3.set_title(f'Price vs Rating (corr: {df_numeric["price"].corr(df_numeric["rating"]):.3f})')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    log_counts = np.log10(df['ratings_count'] + 1)
    sns.kdeplot(x=log_counts, y=df['rating'], cmap='viridis',
                fill=True, levels=10, ax=ax4)
    ax4.set_xlabel('log10(Ratings Count + 1)')
    ax4.set_ylabel('Rating')
    ax4.set_title('2D Density: Rating vs Ratings Count')

    ax5 = fig.add_subplot(gs[2, 1])
    hexbin = ax5.hexbin(df['ratings_count'], df['rating'],
                        gridsize=30, cmap='YlOrRd', mincnt=1, xscale='log')
    ax5.set_xlabel('Ratings Count (log scale)')
    ax5.set_ylabel('Rating')
    ax5.set_title('Hexbin: Point Density')
    plt.colorbar(hexbin, ax=ax5, label='Book Count')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close('all')


def plot_categorical(df, save_path=None): # 4 графика на одном текстовые 
    """
    Categorical analysis by authors and publishers
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Categorical Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0] 
    top_authors = df['author'].value_counts().head(15)
    top_authors.plot(kind='barh', ax=ax1, color=COLORS[0])
    ax1.set_title('Top 15 Authors by Book Count')
    ax1.set_xlabel('Book Count')
    ax1.set_ylabel('Author')
    ax1.invert_yaxis()

    ax2 = axes[0, 1]
    top_publishers = df['publisher'].value_counts().head(15)
    top_publishers.plot(kind='barh', ax=ax2, color=COLORS[1])
    ax2.set_title('Top 15 Publishers by Book Count')
    ax2.set_xlabel('Book Count')
    ax2.set_ylabel('Publisher')
    ax2.invert_yaxis()

    ax3 = axes[1, 0]
    author_stats = df.groupby('author').agg({
        'rating': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'book_count'})
    top_authors_rating = author_stats[author_stats['book_count'] >= 3].nlargest(15, 'rating')
    top_authors_rating['rating'].plot(kind='barh', ax=ax3, color=COLORS[2])
    ax3.set_title('Top 15 Authors by Average Rating (>=3 books)')
    ax3.set_xlabel('Average Rating')
    ax3.set_ylabel('Author')
    ax3.invert_yaxis()
    ax3.set_xlim(3.0, 5.0)

    ax4 = axes[1, 1]
    top_10_publishers = df['publisher'].value_counts().head(10).index
    df_top_pub = df[df['publisher'].isin(top_10_publishers)]
    sns.boxplot(data=df_top_pub, y='publisher', x='rating',
                ax=ax4, palette='Set2', order=top_10_publishers)
    ax4.set_title('Rating Distribution by Top 10 Publishers')
    ax4.set_xlabel('Rating')
    ax4.set_ylabel('Publisher')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # сохраняет в папку


    plt.close('all')



def main():
    """Main function."""
#    script_dir = Path(__file__).parent
    data_file = "popular_filled.csv"


    df = load_data(data_file)

    output_dir = "visualizations"
    output_dir.mkdir(exist_ok=True) # есть ли директория проверка идет иначе ее создаем

   # вызыввает 3 функции с датафраим и созраняет в файл 
    plot_distributions(df, "visualizations/01_distributions.png")

# вызыввает 3 функции с датафраим и созраняет в файл 

    plot_correlations(df, "visualizations/02_correlations.png")

# вызыввает 3 функции с датафраим и созраняет в файл 

    plot_categorical(df, "visualizations/03_categorical.png")
   


if __name__ == "__main__":
    main()
