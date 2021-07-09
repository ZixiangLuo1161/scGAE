# scGAE
We present the single-cell graph autoencoder (scGAE), a dimensionality reduction method that preserves topological structure in scRNA-seq data. scGAE builds a cell graph and uses a multitask-oriented graph autoencoder to preserve topological structure information and feature information in scRNA-seq data simultaneously. We further extended scGAE for scRNA-seq data visualization, clustering, and trajectory inference. Analyses of simulated data showed that scGAE accurately reconstructs developmental trajectory and separates discrete cell clusters under different scenarios, outperforming recently developed deep learning methods. Furthermore, implementation of scGAE on empirical data showed scGAE provided novel insights into cell developmental lineages and preserved inter-cluster distances.

See details in our paper: "scGAE: topology preserving dimensionality reduction for single cell RNA-seq data using graph autoencoder".

Quick start:
Download the model files (preprocessing.py, scgae.py, layers.py, losses.py, and utils.py) in this repository and import functions in them.
1.Prepare datasets
scGAE takes preprocessed data as input. Single cell data preprocessing can be done with Seurat R package and scanpy python package.
2.Model building
Compute the adjacency matrix using get_adj function and build the SCGAE model.
3.Model training
Train the model with model.train() function.
4.Clustering training (optional)
After getting the clustering centers using get_centers_louvain function, perform the clustering training using model.clustering_train() function.
5.Generate embedding
Get the latent embedding of the data by model.embedding().  


Requirements:  
tensorflow --- 2.3.0  
numpy --- 1.18.1  
pandas --- 1.0.1   
tensorflow_probability --- 0.11.0  
scikit-learn --- 0.22.1  
scanpy --- 1.4.4  
anndata --- 0.7.4  
spektral --- 0.6.1  
matplotlib --- 3.0.3  

Usage:  
Running example and parameter setting can be found at Run_scGAE.ipynb

