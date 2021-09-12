
import faiss 
import numpy as np 


def compute_neighbor_accuracy(features, targets, k=4):
    fvecs = np.concatenate(list(features.values()), axis=0)
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs.astype(np.float32))
    _, neighbor_idx = index.search(fvecs, k+1)
    predictions = {name: indices for name, indices in zip(list(features.keys()), neighbor_idx) if name in targets.keys()}
    
    idx2file = {i: file for i, file in enumerate(list(features.keys()))}
    for k, v in predictions.items():
        new_v = [idx2file[idx] for idx in v]
        predictions[k] = new_v
        
    intersection, union = 0, 0
    for file in targets.keys():
        ref, pred = [str(f) for f in targets[file]], [str(f) for f in predictions[file]]
        intersection += len(list(set(ref) & set(pred)))
        union += len(list(set(ref) | set(pred)))
    return intersection/union