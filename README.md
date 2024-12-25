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
The code has six parts, including B-factor prediction, Betti number and barcode, dynamics on EEG data, image classification, RS Plot, and SingleCellDataProcess. Each part has several routines wrote by Python. Please make sure to modify the file paths in the routines to the reader's own paths, and install the Python packages mentioned at the beginning of the routine. Readers can remove the numbers at the beginning of the code when running the program. The details of main purpose of all routines are as following:  
 1. B-factor prediction  
(1) PDB_data_process.py: Loads the three-dimensional coordinates and B-factors for each protein.  
(2) plot_B_prediction.py: Plots the 3D structure of a specific protein, colored by both experimental B-factors and predicted B-factors from different methods.  
(3) slurm_364_B_3_1.py + multi_slurm_364.py: The first code simulates synchronization behavior in a system based on the Lorenz model, updating the state (x, y, z) of atoms by solving dynamic equations, calculating the synchronization index, and analyzing the relationship between the protein's B-factor and the simulation results using a multiple linear regression model. The second code, when updated with the pdbID, can traverse 364 protein datasets and generate job scripts for submission on a Slurm-managed high-performance computing cluster, enabling large-scale simulations and distributed computing analysis.  

2. Betti number and barcode  
(1) persistent_homology_visualizer.py: filtriation process of persistent homology and persistent Laplacian.   
(2) plot_betti_012_reduce.py: give the difference of toplogy of brain neural network with different filitration radius values between 14 healthy controls and 14 schizophreniacs on Betti-0, Betti-1, and Betti-2.  
(3) plot_betti_012.py: give the persistent barcode of Betti-0, Betti-1, and Betti-2.  
(4) plot_betti_reduce_filitration.py: for fig3 e, f, g in main text. it is similar with plot_betti_012_reduce.py.  
(5) plot_betti_time.py: plot the time evolution of Betti-0, Betti-1, and Betti-2 for healthy and schizophrenia.  
(6) plot_EEG_rainbow_butterfly_connective_and_three.py: plot the connectivity matrices, the trajectories of the coupled chaotic socillators, and butterfly wing patterns for systems.  
(7) plot_fig3.py: plot the fig3 in main text.  
(8) plot_phase_eeg_h.py: the phase diagrams of brain neural networks for the healthy controls in the plane of coupling strength and the filtration radius or distance cutoff.  
(9) plot_phase_eeg_s.py: the phase diagrams of brain neural networks for the schizophreniacs in the plane of coupling strength and the filtration radius or distance cutoff.  

3. dynamics on EEG data  
(1) correlation matrix_eeg.py: generate the correlation matrix of EEG data.  
(2) data_eeg_process.py: do the pre-process for the EEG data using Butterworth filter.  
(3) slurm_feature_eeg_mat.py: generate the feature of EEG data and plot the connectivity matrices, the trajectories of the coupled chaotic socillators, and orbit pattern for systems.  

4. image classification  
(1) allaml_classification_pca.py: use PCA method to do the classification prediction of image data.  
(2) allaml_classification_tsne.py: use t-SNE method to do the classification prediction of image data.  
(3) allaml_classification_umap.py: use Umap method to do the classification prediction of image data.  
(4) win_Lorentz_allaml_feature.py: use Lorentz oscillators to generate the features of image data.  
(5) win_rossler_allaml_feature.py: use Rossler oscillators to generate the features of image data.   
(6) win_allaml_feature_CHEN.py: use Chen oscillators to generate the features of image data.  

5. RS Plot  
(1) rs_plot.py:implement the residue-similarity (R-S) analysis for the clustering visualization of classification performance on single cell RNA sequencing (scRNA-seq) data sets.  
(2) rs_score.py: calculate the residue score and the similarity score, which are introduced to evaluate and visualize dimensionality reduction, clustering, and classification algorithms.  
(3) rs_Umap_heatmap_plot.py: R-S plot, Umap, and confusion marix of RS plots of scRNA-seq data sets.  
(4) u-map.py: Umap plots for scRNA-seq data sets.  

6. SingleCellDataProcess  
(1) classification_DBZ_GBDT.py: classification prediction for single cell RNA sequencing with GradientBoostingClassifier and five-fold cross validation.  
(2) classification_DBZ_RF.py: classification prediction for single cell RNA sequencing with RandomForestClassifier and five-fold cross validation.  
(3) classification_DBZ_SVM.py: classification prediction for single cell RNA sequencing with svm algorithm and five-fold cross validation.  
(4) slurm_GSE_feature_4_runge.py: generate features with four-order Runge Kuta algorithm for single cell RNA sequencing data.  
(5) slurm_GSE_feature.py: generate features with one-order forward Euler  algorithm for single cell RNA sequencing data.  

Data Description
======

1. EEG data of the normal category, preictal category, and seizure category  
Another EEG datasets used in Fig.4a of main text is the publicly available EEG dataset collected and curated by Andrzejak et al. from the University of Bonn, Germany. This dataset can be accessed from the official website of the Epileptology Department at the University of Bonn. The normal category (o.zip) comprises single-channel EEG segments recorded from healthy individuals with no history of epilepsy. The preictal category (f.zip) consists of EEG signals collected from epilepsy patients during non-seizure periods. Meanwhile, the seizure category (s.zip) includes EEG signals recorded from the same patients during epileptic seizures. SET B corresponds to o.zip, SET D corresponds to f.zip, and SET E corresponds to s.zip. Each category contains 100 single-channel EEG signals, with each signal having a duration of 23.6 seconds. The sampling frequency of the recordings is 173.61 Hz, resulting in a total of 4097 data points per signal.

2. Image  
The Columbia Object Image Library (COIL-20) is a well-known image dataset created for machine learning
and computer vision research, especially in object classification. Compiled at Columbia University in
1996, it contains images of 20 different objects, each photographed from various angles. Specifically, each
object is captured at 5-degree intervals as it rotates a full 360 degrees, resulting in 72 images per object
and a total of 1,440 images. The dataset includes a variety of everyday items like toys, household goods,
and tools, which offer a wide range of shapes, textures, and colors. This diversity makes COIL-20 highly
useful for developing and testing algorithms that need to generalize across different types of objects and
their appearances.

3. Protein  
B-factor describes how much an atom fluctuate around its mean position in crystal
structures. Protein B-factors quantitatively measure the relative thermal motion of each
atom and reflects atomic flexibility and dynamics. Though B-factor is also affected
by factors such as the refinement methods, it is still a relatively robust measurement
of atomic flexibility in proteins.

4. Single Cell RNA Sequencing  
Single cell RNA sequencing (scRNA-seq) reveals heterogeneity within cell types, leading to an understanding of cell−cell communication, cell differentiation, and differential gene expression. With current technology and protocols, more than 20,000 genes can be identified. Numerous data analysis pipelines have been developed to help analyze such complex data.



Demonstration
======
1. Figure 2  
   In this section, the code and files we used are all stored in figures/Figure 2.

   (1)We obtained Figure 2a by running the persistent_homology_visualizer.py. Each of the sixteen nodes in the regular hexadecagon is injected with a Lorentz oscillator. The coupling or connectivity between these nodes is provided by the radius filtration of the persistent Laplace operator. The three plots display three typical filtration patterns.


   (2) Figures 2e-g show the visualizations of atom_coordinates_1.txt, atom_coordinates_2.txt, and atom_coordinates_3.txt, respectively. The point cloud images were created using PowerPoint, while the dynamic plots were generated by running the MND_filtration_eeg_cat018.py script. Panel e: Folding geometry and synchronized dynamics of a 120-element point cloud. Panel f: Partial folding geometry and partial synchronized dynamics of a 120-element point cloud. Panel g: Unfolded geometry and chaotic dynamics of a 120-element point cloud.

  
2. Figure 4  
In this section, the code and files we used are all stored in figures/Figure 4. We saved the three-dimensional coordinates of the data points in topologicaldynamics.xyz and obtained Figures 4d-e by running the rips.py script. Panel d illustrates the filtration process of the point cloud, generating a sequence of simplicial complexes. Panel e shows the variation of the persistent Betti numbers as the filtration radius increases.


Download the repository
======
```
  # download repository by git  
  git clone https://github.com/kelu0124/TEPC.git
```
License
======
All codes released in this study is under the MIT License.
