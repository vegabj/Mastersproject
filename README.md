# Mastersproject
Machine learning classifiers for different biases in miRNA datasets.
* Python 3.6
* R 3.5.2


## Requirements Python
Note: Anaconda is recommended for Windows users.
* Numpy
* Pandas
* Scikit-learn
* Seaborn
* Graphviz
* gseapy
* tqdm
* R
* rpy2

## Requirements R
* limma
* edgeR
* statmod


## Usage
### Create enrichment scores
* generate_enrichment_score - Creates enrichment scores for a given dataset - This requires a GMT file
* create_gmt - Creates a gmt file for a given dataset, see source code for instructions for non RNA-sequencing sets. The gmt files should be combined.

### Create score spreadsheets
* generate_score_sheet - Creates a score sheet for selected data sets
* generate_score_sheet_es - Creates a score sheet for selected data sets based on enrichment score

### Plots
* pca - Creates a PCA plots over selected data sets.
* visualize_decision_tree - visualizes a decision trees as pdf files
* dual_heatmap - Creates heatmaps from two selected score sheets that are latex friendly
* analyze_score_sheet - Creates a heatmap from a selected score sheet
* print_feature_importance - Prints the feature importance in both SVM and Random Forest and makes a plot for the top 20 features in SVM.
