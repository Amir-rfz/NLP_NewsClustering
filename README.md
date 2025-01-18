## NLP_NewsClustering

### Overview
Clustering is an unsupervised learning technique used to group similar data points. This project focuses on clustering news articles, aiming to group them into distinct clusters based on their content. By using advanced text embeddings and clustering algorithms, this project demonstrates the practical applications of clustering in natural language processing (NLP).

### Features

- **Text Preprocessing**: Includes tokenization, stop-word removal, and lemmatization/stemming.
- **Feature Extraction**: Uses SentenceTransformers (all-MiniLM-L6-v2) to generate high-quality embeddings of textual data.
- **Clustering Algorithms**:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering
- **Dimensionality Reduction**: Implements PCA for visualization and better cluster representation.
- **Evaluation Metrics**:
  - Silhouette Score
  - Homogeneity
- **Visualizations**: Generates charts to illustrate clustering results and comparisons.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Amir-rfz/NLP_NewsClustering.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NLP_NewsClustering
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook clustering.ipynb
   ```
2. Follow the steps in the notebook to:
   - Preprocess the data
   - Extract features
   - Apply clustering algorithms
   - Evaluate and visualize the results

### Data

The dataset used contains a collection of English news articles. Ensure the dataset is available in the correct format before running the notebook. The default dataset file is `articles.csv` under the `data/` directory.

### Key Algorithms and Libraries

- **Algorithms**:
  - K-Means Clustering
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Hierarchical Clustering
- **Libraries**:
  - `scikit-learn`
  - `sentence-transformers`
  - `numpy`
  - `matplotlib`
  - `pandas`

### Results

#### Algorithm Comparison
- **Silhouette Scores**: Highlights the quality of clusters formed by different algorithms.
- **Homogeneity**: Measures the uniformity of cluster content.

#### Visualizations
- PCA-based 2D projections of clusters.
- Comparison charts for clustering performance.

### File Structure

```
NLP_NewsClustering/
|— clustering.ipynb         # Jupyter Notebook with code and analysis
|— requirements.txt         # List of required Python libraries
|— data/                    # Directory for datasets
    |— articles.csv         # Sample dataset
```

### Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request with your enhancements.

