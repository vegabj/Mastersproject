# Mastersproject
Machine learning classifier for combined microRNA datasets with different biases.
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
* print_feature_importance - Prints the feature importance in both SVM and Random Forest, makes a plot for the top 20 features in SVM, creates a scatter plot over SVM and Random Forest feature importance.
* roc_rf - Creates a ROC curve for selected data sets and scaling using Random Forest.
* roc_svm - Creates a ROC curve for selected data sets and scaling using SVM.
* box_plot - Creates a box plot of miRNAs.
* density_plot - Creates a density plot of selected data sets.

## Setup
* A /Data/ folder must be created to hold data sets.
* A /Out/ folder must be created to hold generated data such as score spreadsheets.
* A /Plots/ folder must be created to store generated plots.
