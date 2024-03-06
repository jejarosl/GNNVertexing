#!/usr/bin/env python3

import time
import glob
import uproot
import numpy as np
import awkward as ak
import pickle
from tqdm import tqdm

import torch
from torch_geometric.data import Data

THRESHOLD_Z = 0.3

def mkdir_p(mypath):
    '''Function to create a new directory, if it not already exist.
       Args:
           mypath : directory path
    '''
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:
        if not (exc.errno == EEXIST and path.isdir(mypath)):
            raise

def computeEdgeAndLabels(track_data, edges, edges_labels, edge_features):
    '''Compute the truth graph'''
    print("Building the graph")

    n_tracks = len(track_data.gnn_pt)
    z_pca = np.array(track_data.gnn_z_pca)
    sim_vertex_ID = np.array(track_data.gnn_sim_vertex_ID)
    gnn_pt = np.array(track_data.gnn_pt)
    t_pi = np.array(track_data.gnn_t_pi)
    t_k = np.array(track_data.gnn_t_k)
    t_p = np.array(track_data.gnn_t_p)
    dz = np.array(track_data.gnn_dz)

    for i in tqdm(range(n_tracks)):
        z_diff = np.abs(z_pca[i+1:] - z_pca[i])
        ids = np.where(z_diff < THRESHOLD_Z)[0]
        indices = ids + i + 1

        edges.extend([(j, i) for j in indices])

        labels = (sim_vertex_ID[i] == sim_vertex_ID[indices]).astype(int)
        edges_labels.extend(labels)
        
        pt_diff = np.abs(gnn_pt[indices] - gnn_pt[i])
        time_comps = np.array([np.abs(t_pi[indices] - t_pi[i]),
                      np.abs(t_k[indices] - t_k[i]),
                      np.abs(t_p[indices] - t_p[i]),
                      
                      np.abs(t_pi[indices] - t_k[i]),
                      np.abs(t_pi[indices] - t_p[i]),
                      
                      np.abs(t_k[indices] - t_pi[i]),
                      np.abs(t_k[indices] - t_p[i]),
                      
                      np.abs(t_p[indices] - t_k[i]),
                      np.abs(t_p[indices] - t_pi[i])])
                      
        z_diff = z_diff[ids]
        
        for k, ind in enumerate(indices):
            dz_significance = (z_pca[i] - z_pca[ind]) / np.sqrt(dz[i]**2 + dz[ind]**2)
            edge_features.append([pt_diff[k], z_diff[k], dz_significance] + list(time_comps[:, k]))
            
def set_small_to_zero(a, eps=1e-8):
    a[np.abs(a) < eps] = 0
    return a

def remap_PIDs(pids):
    """
    Remaps particle IDs to a simplified classification scheme.
    
    Args:
    - pids (list): List of particle IDs to be remapped.
    
    Returns:
    - remapped_pids (list): List of remapped particle IDs.
    """
    # Mapping of particle IDs to simplified classification
    pid_map = {11: 0, 13: 0, 211: 0, 321: 1, 2212: 2, 3112: 2}
    
    # Remap PIDs using the pid_map dictionary
    remapped_pids = [pid_map.get(pid, -1) for pid in pids]
    
    return remapped_pids

def process_files(input_folder, output_folder, n_files=100000, offset=0):
    files = glob.glob(f"{input_folder}/*.root")
    print(f"Number of files: {len(files)}")

    X, Edges, Edges_labels, Edge_features = [], [], [], []
    PIDs_truth, Times_truth = [], []

    mkdir_p(output_folder)

    for i_file, file in enumerate(files[offset:offset+n_files]):
        i_file += offset
        
        print('\nProcessing file {} '.format(file))
        try:
            with uproot.open(file) as f:
                tree = f["vertices4DValid"]
                
                for ev, key in enumerate(tree):
                
                    t = tree[key]
                    track_data = t.arrays(["gnn_weight", "gnn_pt", "gnn_eta", "gnn_phi", "gnn_z_pca",
                                            "gnn_t_pi", "gnn_t_k", "gnn_t_p", "gnn_mva_qual", 'gnn_btlMatchChi2',
                                            'gnn_btlMatchTimeChi2', 'gnn_etlMatchChi2', "gnn_sim_vertex_ID",
                                            'gnn_etlMatchTimeChi2', 'gnn_pathLength', 'gnn_npixBarrel', 'gnn_npixEndcap',
                                            'gnn_mtdTime', 'gnn_is_matched_tp', 'gnn_dz', 'gnn_sigma_t0safe'])
                    truth_data = t.arrays(['gnn_tp_tsim', 'gnn_tp_tEst', 'gnn_tp_pdgId'])

                    number_of_tracks = len(track_data.gnn_weight)
                    print(f"{i_file}_{ev} : Have {number_of_tracks} tracks in the file")
                    
                    start = time.time()

                    x_ev = np.array([track_data.gnn_weight,
                                     track_data.gnn_pt,
                                     track_data.gnn_eta,
                                     track_data.gnn_phi,
                                     track_data.gnn_z_pca,
                                     track_data.gnn_t_pi,
                                     track_data.gnn_t_k,
                                     track_data.gnn_t_p,
                                     track_data.gnn_mva_qual,
                                     track_data.gnn_btlMatchChi2,
                                     track_data.gnn_btlMatchTimeChi2,
                                     track_data.gnn_etlMatchChi2,
                                     track_data.gnn_etlMatchTimeChi2,
                                     track_data.gnn_pathLength,
                                     track_data.gnn_npixBarrel,
                                     track_data.gnn_npixEndcap,
                                     track_data.gnn_mtdTime,
                                     track_data.gnn_dz,
                                     track_data.gnn_sigma_t0safe], 
                                    dtype=np.float32)

                    x_ev = set_small_to_zero(x_ev, eps=1e-5)

                    print(f"{i_file}_{ev} : Got the track properties")

                    X.append(x_ev)

                    edges, edges_labels, edge_features = [], [], []
                    pids, times = truth_data.gnn_tp_pdgId, truth_data.gnn_tp_tEst

                    # Call the function to compute edges and labels
                    computeEdgeAndLabels(track_data, edges, edges_labels, edge_features)
                    
                    print(len(edge_features), len(edge_features[0]))

                    Edges.append(np.array(edges).T)
                    PIDs_truth.append(np.array(pids, dtype=np.int64))
                    Times_truth.append(np.array(times, dtype=np.float32))
                    Edges_labels.append(np.array(edges_labels))
                    Edge_features.append(np.array(edge_features))
                    
                    
                    if (ev % 10 == 0 and ev != 0) or ev == len(tree.keys())-1:
                        stop = time.time()
                        print(f"t = {stop - start} ... Saving the pickle data for {i_file}_{ev}")

                        # Save the processed data into pickle files
                        with open(f"{output_folder}{i_file}_{ev}_node_features.pkl", "wb") as fp:
                            pickle.dump(X, fp)
                        with open(f"{output_folder}{i_file}_{ev}_edges.pkl", "wb") as fp:
                            pickle.dump(Edges, fp)
                        with open(f"{output_folder}{i_file}_{ev}_edges_labels.pkl", "wb") as fp:
                            pickle.dump(Edges_labels, fp)
                        with open(f"{output_folder}{i_file}_{ev}_edge_features.pkl", "wb") as fp:
                            pickle.dump(Edge_features, fp)
                        with open(f"{output_folder}{i_file}_{ev}_times_truth.pkl", "wb") as fp:
                            pickle.dump(Times_truth, fp)
                        with open(f"{output_folder}{i_file}_{ev}_PID_truth.pkl", "wb") as fp:
                            pickle.dump(PIDs_truth, fp)
                            
                        X, Edges, Edges_labels, Edge_features = [], [], [], []
                        PIDs_truth, Times_truth = [], []
                        start = time.time()


        except Exception as e:
            print(f"Error: {e}")
            continue


def loadData(path, num_files = -1):
    """
    Loads pickle files of the graph data for network training.
    """
    f_edges_label = glob.glob(f"{path}*edges_labels.pkl")
    f_edges_features = glob.glob(f"{path}*edge_features.pkl")
    f_edges = glob.glob(f"{path}*edges.pkl" )
    f_nodes_features = glob.glob(f"{path}*node_features.pkl")
    f_PID = glob.glob(f"{path}*PID_truth.pkl")
    f_times = glob.glob(f"{path}*times_truth.pkl")
    
    
    edges_label, edges, nodes_features, edges_features, PID_truth, times_truth = [], [], [], [], [], []
    n = len(f_edges_label) if num_files == -1 else num_files

    for i_f, _ in enumerate(tqdm(f_edges_label)):
        
        # Load the data
        if (i_f <= n):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
                
            f = f_edges_features[i_f]
            with open(f, 'rb') as fb:
                edges_features.append(pickle.load(fb))
                
            f = f_edges[i_f]
            with open(f, 'rb') as fb:
                edges.append(pickle.load(fb))
                
            f = f_nodes_features[i_f]
            with open(f, 'rb') as fb:
                nodes_features.append(pickle.load(fb))
                
            f = f_PID[i_f]
            with open(f, 'rb') as fb:
                PID_truth.append(pickle.load(fb))
                
            f = f_times[i_f]
            with open(f, 'rb') as fb:
                times_truth.append(pickle.load(fb))
                
        else:
            break
            
    return edges_label, edges, nodes_features, edges_features, PID_truth, times_truth


def prepare_test_data(data_list, ev):
    """
    Function to prepare (and possibly standardize) the test data
    """
    x_np, edge_label, edge_index, edge_features, PIDs, Times = data_list[ev]
    #x_norm, mean, std = standardize_data(x_np)

    # Create torch vectors from the numpy arrays
    x = torch.from_numpy(x_np)
    x = torch.nan_to_num(x, nan=0.0)
    
    e_label = torch.from_numpy(edge_label)
    edge_index = torch.from_numpy(edge_index)
    e_features = torch.from_numpy(edge_features)
    e_PIDs = torch.from_numpy(PIDs)
    e_times = torch.from_numpy(Times)
    
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=edge_label,
               edge_features=edge_features, e_PIDs=e_PIDs, e_times=e_times)
    return data


def flatten_lists(el, ed, nd, ef, pid, times):
    edge_label, edges, node_data, edge_features, PID_truth, times_truth = [], [], [], [], [], []
    for i, X in enumerate(nd):
        for ev in range(len(X)):
                  
            if len(ed[i][ev]) == 0:
                print(f"Event {i}:{ev} has NO edges. Skipping.")
                continue # skip events with no edges
                
            elif X[ev].shape[1] <= 1:
                print(f"Event {i}:{ev} has {X[ev].shape[1]} nodes. Skipping.")
                continue
            else:
                edges.append(ed[i][ev])
                edge_label.append(el[i][ev])
                node_data.append(X[ev])
                edge_features.append(ef[i][ev])
                PID_truth.append(pid[i][ev])
                times_truth.append(times[i][ev])
                
    return edge_label, edges, node_data, edge_features, PID_truth, times_truth


def save_dataset(pickle_data, output_location, trainRatio = 0.8, valRatio = 0.1, testRatio = 0.1, num_files=-1):
    
    print("Loading Pickle Files...")
    # obtain edges_label, edges, nodes_features... from all the pickle files
    el, ed, nd, ef, pid, times = loadData(pickle_data, num_files = num_files)
    print("Loaded.")

    edge_label, edge_data, node_data, edge_features, PIDs, Times = flatten_lists(el, ed, nd, ef, pid, times)

    data_list = []
    print(f"{len(node_data)} total events in dataset.")

    nSamples = len(node_data)
    nTrain = int(trainRatio * nSamples)
    nVal = int(valRatio * nSamples)

    print("Preparing training and validation split")
    for ev in tqdm(range(len(node_data[:nTrain+nVal]))):
                
        x_np = node_data[ev].T
        #x_norm, _, _ = standardize_data(x_np)
        
        # Create torch vectors from the numpy arrays
        x = torch.from_numpy(x_np)
        x = torch.nan_to_num(x, nan=0.0)
        
        e_label = torch.from_numpy(edge_label[ev])
        edge_index = torch.from_numpy(edge_data[ev])
        e_features = torch.from_numpy(edge_features[ev])
        e_PIDs = torch.from_numpy(PIDs[ev])
        e_times = torch.from_numpy(Times[ev])

        data = Data(x=x, num_nodes=torch.tensor(x.shape[0]),
                    edge_index=edge_index, edge_label=e_label, 
                    edge_features=e_features, e_PIDs=e_PIDs, e_times=e_times)
        
        # This graph is directed.
        #print(f"data is directed: {data.is_directed()}")
        data_list.append(data)

    # The test split is not normalized and is stored as a list
    test_data_list = []
    
    print("Preparing test split (data not preprocessed)")
    for ev in tqdm(range(len(node_data[nTrain+nVal:]))):

        x_np = node_data[ev].T
        # Do not pre-process the test split
        data = [x_np, edge_label[ev], edge_data[ev], edge_features[ev], PIDs[ev], Times[ev]]
        test_data_list.append(data)


    trainDataset = data_list[:nTrain] # training dataset
    valDataset = data_list[nTrain:]   # validation dataset
    
    # Saves the dataset objects to disk.
    mkdir_p(f'{output_location}')
    torch.save(trainDataset, f'{output_location}/dataTraining.pt')
    torch.save(valDataset, f'{output_location}/dataVal.pt')
    torch.save(test_data_list, f'{output_location}/dataTest.pt')
    print("Done: Saved the training datasets.")