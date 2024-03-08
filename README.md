# Repository for 4D vertexing using GNN.

Repository for 4D vertexing for CMS Phase-2 using machine learning. Performs tracks clustering and predicts the track time and PID. 
Based on GNN developed for HGCAL trackster linking (Jekaterina Jaroslavceva <jarosjek@fel.cvut.cz> <jejarosl@cern.ch>, Felice Pantaleo <felice.pantaleo@cern.ch>).

# Dataset and graph building
Sample: TTbar14TeV+PU200 (14_0_0_pre3).
Save track and vertex features after a pre-clustering in 3D and track filtering. 

## Prepare dataset 
In ```DataPreprocessing.ipynb ```. Select node and edge features.    
Node features (19):
- track weight (wrt best-associated reco vertex)
- track pt
- track eta
- track phi
- track z PCA
- track dz
- track time MTD
- track dt
- track time Pi hypothesis
- track time K hypothesis
- track time P hypothesis
- track MVA quality
- track btlMatchChi2
- track btlMatchTimeChi2
- track etlMatchChi2
- track etlMatchTimeChi2
- track pathLength
- track npixBarrel
- track npixEndcap

Edge feature (12):
- pt difference
- z difference
- z difference in units of dz
- time difference for all possible hypothesis (9 combinations)  

Connect nodes if z distance is < 3 mm  

Edge labels: tracks from the same sim vertex  
Time truth: match reco track to TP and evaluate TP time   
PID truth: match reco track to TP and use TP PID 

Remap track PIDs: define three classes Pi, K, P and remap any other particle type to these three, example: ele and mu to Pi, baryons to P.   

Save node features, edge features, edge labels, time truth and PID truth as pickle data.  

## Prepare dataset for training
Split data into train, validation, test sets in the ratio 0.8, 0.1, 0.1   
Create torch vectors from the numpy arrays.  


# GNN Model & Training
In ```GNN_training.ipynb ```. Trained with PyTorch.
Load dataset and prepare for training (```DataLoader```). Initialize weights with Xavier initialization.  

## GNN_TrackLinkingNet
GNN Model:  
1. Transformation to latent space, one for node features and one for edge features. Fully connected NN with LeakyReLU activation and dropout.
2. Edge attention (direct and reverse), with LeakyReLu for input layer and sigmoid for output layer
3. EdgeConv blocks for neighbourhood aggregation (2 blocks).
4. Edge classification, NN  with LeakyReLu for input layer and sigmoid for output  

## EdgeConvBlock

## Training
Loss: Focal loss for clustering, MSE for time regression, cross-entropy for PID classification.  
Adam optimizer, CosineAnnealingLR scheduler to to adjust learning rate.  
Performance plots and save model for each epoch.  

# MLP Model
Same as GNN but without EdgeConv.
