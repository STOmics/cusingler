import sys

import h5py
from cusingler import cusingler
import numpy as np

def load_matrix(filename):
    with h5py.File(filename) as fh:
        group = fh["/X"]
        shape = group.attrs["shape"]
        height, width = shape[0], shape[1]
        data = group["data"][...]
        indices = group["indices"][...]
        indptr = group["indptr"][...]
    return height, width, data, indices, indptr

def load_celltypes(filename):
    with h5py.File(filename) as fh:
        is_dataset = False
        if "/obs/celltype" in fh:
            group = fh["/obs/celltype"]
        elif "/obs/ClusterName" in fh:
            group = fh["/obs/ClusterName"]
        elif "/obsm/annotation_au/celltype" in fh:
            is_dataset = True
            temp = fh["/obsm/annotation_au/celltype"][...]
            d = {}
            codes = []
            celltypes = []
            for t in temp:
                if t not in d:
                    d[t] = len(d)
                    codes.append(t)
                celltypes.append(d[t])
        else:
            raise("No celltype found.")
        
        if not is_dataset:
            codes = group["codes"][...]
            celltypes = group["categories"][...]
    return codes, celltypes

def load_cellnames(filename):
    with h5py.File(filename) as fh:
        cellnames = fh["/obs/_index"][...]
    return cellnames

def load_geneidx(filename):
    with h5py.File(filename) as fh:
        group = fh["/var"]
        name = group.attrs["_index"]
        if name == "Gene_ID":
            name = "Symbol"
        return group[name][...]

def main(ref_file, qry_file, out_file):
    ref_height, ref_width, ref_data, ref_indices, ref_indptr = load_matrix(ref_file)
    qry_height, qry_width, qry_data, qry_indices, qry_indptr = load_matrix(qry_file)

    codes, celltypes = load_celltypes(ref_file)
    cellnames = load_cellnames(qry_file)
    ref_geneidx = load_geneidx(ref_file)
    qry_geneidx = load_geneidx(qry_file)

    cores = 20
    gpuid = 0
    res = cusingler.run(cores, gpuid, ref_height, ref_width, ref_data, ref_indices, ref_indptr,
                  qry_height, qry_width, qry_data, qry_indices, qry_indptr,
                  codes, celltypes,
                  cellnames, ref_geneidx, qry_geneidx, quantile=80, finetune_thre=0.05, finetune_times=0)
    arr = np.array(res)
    np.savetxt(out_file, arr, delimiter='\t', header='cell\tfirstLabel\tfinalLabel',  
                comments="", fmt='%s')
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("enter <ref file> <qry file> <out stat file>")
        sys.exit(-1)

    ref_file = sys.argv[1]
    qry_file = sys.argv[2]
    out_file = sys.argv[3]


    main(ref_file, qry_file, out_file)