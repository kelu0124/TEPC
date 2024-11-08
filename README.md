# TEPC
The TEPC (topology-enabled predictions from chaos) platform that provides a general methodology of topology-enabled machine learning (ML) predictions from chaotic dynamics.
 
Requirements
======
Platform Requirements


<ul>
 <li>A job scheduling platform such as SLURM or an equivalent resource management system for distributed computing.
SLURM must be properly configured on the system for managing job submissions and execution.</li>  
  <li>SLURM must be properly configured on the system for managing job submissions and execution.</li>
</ul>

Python Dependencies  
<ul>
 <li>Python==3.8.2 </li>
 <li>setuptools (>=18.0) </li>
 <li>numpy (1.17.4) (>=18.0) </li>
 <li>scikit-learn (0.23.2) </li>
 <li>scipy (1.11.4) </li>
 <li>pandas (2.1.4) </li>
 <li>openpyxl (3.1.2) </li>
</ul>

Code Description
======
The code has six parts, including B-factor prediction, Betti number and barcode, dynamics on EEG data, image classification, RS Plot, and single cell. Each part has several routines wrote by Python. Please make sure to modify the file paths in the routines to the reader's own paths, and install the Python packages mentioned at the beginning of the routine. The details of main purpose of all routines are as following:  
 1. B-factor prediction  
(1) PDB_data_process.py: Loads the three-dimensional coordinates and B-factors for each protein.  
(2) plot_B_prediction.py: Plots the 3D structure of a specific protein, colored by both experimental B-factors and predicted B-factors from different methods. 
(3) slurm_364_B_3_1.py + multi_slurm_364.py: The first code simulates synchronization behavior in a system based on the Lorenz model, updating the state (x, y, z) of atoms by solving dynamic equations, calculating the synchronization index, and analyzing the relationship between the protein's B-factor and the simulation results using a multiple linear regression model. The second code, when updated with the pdbID, can traverse 364 protein datasets and generate job scripts for submission on a Slurm-managed high-performance computing cluster, enabling large-scale simulations and distributed computing analysis.


3. Betti number and barcode  
(1) persistent_homology_visualizer.py: filtriation process of persistent homology and persistent Laplacian.   
(2) plot_betti_012_reduce.py: give the difference of toplogy of brain neural network with different filitration radius values between 14 healthy controls and 14 schizophreniacs on Betti-0, Betti-1, and Betti-2.  
(3) plot_betti_012.py: give the persistent barcode of Betti-0, Betti-1, and Betti-2.  
(4) plot_betti_reduce_filitration.py: for fig3 e, f, g in main text. it is similar with plot_betti_012_reduce.py.  
(5) plot_betti_time.py: plot the time evolution of Betti-0, Betti-1, and Betti-2 for healthy and schizophrenia.  
(6) plot_EEG_rainbow_butterfly_connective_and_three.py: plot the connectivity matrices, the trajectories of the coupled chaotic socillators, and butterfly wing patterns for systems.  
(7) plot_fig3.py: plot the fig3 in main text.  
(8) plot_phase_eeg_h.py: the phase diagrams of brain neural networks for the healthy controls in the plane of coupling strength and the filtration radius or distance cutoff.  
(9) plot_phase_eeg_s.py: the phase diagrams of brain neural networks for the schizophreniacs in the plane of coupling strength and the filtration radius or distance cutoff.  

4. dynamics on EEG data  
(1) correlation matrix_eeg.py: generate the correlation matrix of EEG data.  
(2) data_eeg_process.py: do the pre-process for the EEG data using Butterworth filter.  
(3) slurm_feature_eeg_mat.py: generate the feature of EEG data and plot the connectivity matrices, the trajectories of the coupled chaotic socillators, and orbit pattern for systems.  

5. image classification  
(1) allaml_classification_pca.py: use PCA method to do the classification prediction of image data.  
(2) allaml_classification_tsne.py: use t-SNE method to do the classification prediction of image data.  
(3) allaml_classification_umap.py: use Umap method to do the classification prediction of image data.  
(4) win_Lorentz_allaml_feature.py: use Lorentz oscillators to generate the features of image data.  
(5) win_allaml_feature_CHEN.py: use Chen oscillators to generate the features of image data.  

6. RS Plot  
(1) rs_plot.py:implement the residue-similarity (R-S) analysis for the clustering visualization of classification performance on single cell RNA sequencing (scRNA-seq) data sets.  
(2) rs_score.py: calculate the residue score and the similarity score, which are introduced to evaluate and visualize dimensionality reduction, clustering, and classification algorithms.  
(3) rs_Umap_heatmap_plot.py: R-S plot, Umap, and confusion marix of RS plots of scRNA-seq data sets.  
(4) u-map.py: Umap plots for scRNA-seq data sets.  

7. single cell  
(1) classification_DBZ_GBDT.py: classification prediction for single cell RNA sequencing with GradientBoostingClassifier and five-fold cross validation.  
(2) classification_DBZ_RF.py: classification prediction for single cell RNA sequencing with RandomForestClassifier and five-fold cross validation.  
(3) classification_DBZ_SVM.py: classification prediction for single cell RNA sequencing with svm algorithm and five-fold cross validation.  
(4) slurm_GSE_feature_4_runge.py: generate features with four-order Runge Kuta algorithm for single cell RNA sequencing data. 
(5) slurm_GSE_feature.py: generate features with one-order forward Euler  algorithm for single cell RNA sequencing data.  

Code Description
======
1. EEG
   (1)
   (2)
3. Image
4. Protein
5. Single Cell RNA Sequencing
   

Download the repository
======
```
  # download repository by git  
  git clone https://github.com/kelu0124/TEPC.git
```
License
======
All codes released in this study is under the MIT License.
