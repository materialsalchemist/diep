from __future__ import annotations
import dgl
import torch
import warnings

import numpy as np
from matgl import device

print("Using device:", device)
torch.set_default_device(device)

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

def get_pair_vector_from_graph(g: dgl.DGLGraph):
    bond_vec = torch.zeros(g.num_edges(), 3, dtype=torch.double)
    bond_vec[:, :] = (
        g.ndata["pos"][g.edges()[1][:].long(), :]
        + torch.squeeze(torch.matmul(g.edata["pbc_offset"].unsqueeze(1), torch.squeeze(g.edata["lattice"])))
    )

    return g.ndata["pos"][g.edges()[0][:].long(), :].double(),\
            bond_vec,\
            g.ndata["node_type"][g.edges()[0][:].long()],\
            g.ndata["node_type"][g.edges()[1][:].long()]


def extract_vectors_from_graph(g, triple_bond_indices):
    # import json
    # print('extract_vectors_from_graph')
    # f=open('triple_bond_indices.json','w')
    # json.dump(triple_bond_indices,f)
    # f.close()

    pair_vectors_i, pair_vectors_j, atom_i, atom_j = get_pair_vector_from_graph(g)
    R1 = pair_vectors_i[triple_bond_indices[:,0]]
    R2 = pair_vectors_j[triple_bond_indices[:,0]]
    R3 = pair_vectors_i[triple_bond_indices[:,1]]

    Z1 = atom_i[triple_bond_indices[:,0]]
    Z2 = atom_j[triple_bond_indices[:,0]]
    Z3 = atom_i[triple_bond_indices[:,1]]

    indices_mask = torch.all(torch.eq(R3,R1), dim=1)
    R3[indices_mask,:] = pair_vectors_j[triple_bond_indices[:,1]][indices_mask,:]
    Z3[indices_mask] = atom_j[triple_bond_indices[:,1]][indices_mask]
    return R1, R2, R3, Z1, Z2, Z3

def triangle_transform_cpu(R1,R2,R3,Z1,Z2,Z3):
    # import time
    # start = time.time()
    num_triplets = len(R1)
    r12 = R1-R2
    d12 = torch.sqrt((r12*r12).sum(1))
    r13 = R1-R3
    d13 = torch.sqrt((r13*r13).sum(1))
    r23 = R3 - R2
    d23 = torch.sqrt((r23*r23).sum(1))
    
    # print(R1-R2)
    # print(d12)
    # print(d12.shape)
    distances = torch.stack([d12,d13,d23],axis=1).to(device=torch.device('cpu')) # batch x 3
    # print(distances.shape)
    ordered_distances, order_indices = torch.sort(distances, dim = 1)
    # Z_array = torch.from_numpy(np.array([Z1, Z2, Z3])) # 3 x batch
    Z_array = torch.stack([Z1, Z2, Z3],dim=1) # 3 x batch

    Z_idx = torch.from_numpy(np.array([[0,1,0,2,1,2]])).repeat(num_triplets,1).to(device=torch.device('cpu'))# batch x 6
    d1 = ordered_distances[:, 2].cpu().detach().numpy()
    d2 = ordered_distances[:, 1].cpu().detach().numpy()
    d3 = ordered_distances[:, 0].cpu().detach().numpy()

    # print(torch.stack([order_indices[:,2] * 2,order_indices[:,2] * 2+1],dim=1))
    ZZ12a = Z_idx.gather(1, torch.stack([order_indices[:,2] * 2,order_indices[:,2] * 2+1],dim=1)) # batch x 2
    ZZ13b = Z_idx.gather(1, torch.stack([order_indices[:,0] * 2,order_indices[:,0] * 2+1],dim=1))

    ZZ1arr = torch.where(ZZ12a[:,0]==ZZ13b[:,0],ZZ12a[:,0],ZZ12a[:,1])
    ones_ZZ1 = torch.ones(num_triplets).to(device=torch.device('cpu'))
    ZZ1arr = torch.where((ZZ12a[:,0]!=ZZ13b[:,0]) & (ZZ12a[:,1]!=ZZ13b[:,1]),ones_ZZ1,ZZ1arr)
    ZZ1 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ1arr.type(torch.int64),dim=1)).squeeze()
    ZZ2idx = torch.where(ZZ12a[:,0]==ZZ1arr,ZZ12a[:,1],ZZ12a[:,0])
    ZZ2 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ2idx.type(torch.int64),dim=1)).squeeze()
    ZZ3idx = torch.where(ZZ13b[:,0]==ZZ1arr,ZZ13b[:,1],ZZ13b[:,0])
    ZZ3 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ3idx.type(torch.int64),dim=1)).squeeze()
    r1 = np.zeros([num_triplets, 2],dtype=np.float32)
    r2 = np.zeros([num_triplets, 2],dtype=np.float32)
    r2[:, 0] = d1

    r3_x = 1 / 2 * (d1 + (d3**2 - d2**2) / d1)
    r3_y = np.sqrt(d3**2 - r3_x**2)
    try:
        f = np.vectorize(lambda k: k if not np.isnan(k) else 0.0)
        r3_y = f(r3_y).astype(np.float32) # If d1 = d2 + d3, then just set r3_y to zero, it's a straightline
    except Exception as eee:
        print('r3_y',r3_y)
        print('d3',d3)
        print('r3_x',r3_x)
        print(eee)
    r3 = np.array([r3_x, r3_y]).T
    r_center = (r1 + r2 + r3) / 3

    fR1 = torch.from_numpy(r1-r_center)
    fR2 = torch.from_numpy(r2-r_center)
    fR3 = torch.from_numpy(r3-r_center)
    fR1 = fR1.to(device=torch.device('cpu'))
    fR2 = fR2.to(device=torch.device('cpu'))
    fR3 = fR3.to(device=torch.device('cpu'))
    return fR1,fR2,fR3,ZZ1,ZZ2,ZZ3

def integration_cpu(tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum=True):
    grid_size = 5
    grid = [[(x, y) for x in range(-grid_size, grid_size + 1)] for y in range(-grid_size, grid_size + 1)]
    grid = torch.from_numpy(np.reshape(np.array(grid,dtype=np.float32), (-1,2))).to(device=torch.device('cpu'))

    batchlen = len(tR1)
    grid_batch = torch.unsqueeze(grid, dim=1)
    grid_batch = grid_batch.repeat(1, len(tR1),1)

    print("diep:Performing integration")
    tR1_batch = torch.unsqueeze(tR1, dim=0)
    tR1_batch = tR1_batch.repeat(121,1,1)
    tR2_batch = torch.unsqueeze(tR2, dim=0)
    tR2_batch = tR2_batch.repeat(121,1,1)
    tR3_batch = torch.unsqueeze(tR3, dim=0)
    tR3_batch = tR3_batch.repeat(121,1,1)

    d1_batch  = torch.norm(tR1_batch - grid_batch,dim=2)
    d2_batch  = torch.norm(tR2_batch - grid_batch,dim=2)
    d3_batch  = torch.norm(tR3_batch - grid_batch,dim=2)

    n = torch.exp(-torch.norm(tR1_batch - grid_batch,dim=2)**2) + torch.exp(-torch.norm(tR2_batch - grid_batch,dim=2)**2)
    
    
    tZ1_batch = torch.unsqueeze(tZ1, dim=0).repeat(121,1)
    tZ2_batch = torch.unsqueeze(tZ2, dim=0).repeat(121,1)
    tZ3_batch = torch.unsqueeze(tZ3, dim=0).repeat(121,1)
    # print('tZ1',tZ1.shape)
    # print('n',n.shape)
    zeros = torch.zeros(121, batchlen)
    zeros.to(device=torch.device('cpu'))
    ones = torch.ones(121, batchlen)
    ones.to(device=torch.device('cpu'))
    
    d_batch = n*tZ1_batch/torch.sqrt(d1_batch**2+1) + n*tZ2_batch/torch.sqrt(d2_batch*2+1) + n*tZ3_batch/torch.sqrt(d3_batch*2+1)
    # print('d_batch',d_batch.shape)
    if do_sum:
        d_sums = torch.sum(d_batch, dim=0)
        d_sums = torch.nn.functional.normalize(d_sums,dim=0)
        return torch.reshape(d_sums, (-1,1))
    else:
        d_sums = torch.nn.functional.normalize(d_batch,dim=0)
        # print('d_sums',d_sums.shape)
        return torch.transpose(torch.squeeze(d_sums),0,1)

def calculate_dsum_cpu(grid, tR1, tR2, tR3, num_points_square, tZ1, tZ2, tZ3, do_sum=True,max_n=1):
    batchlen = len(tR1)
    grid_batch = torch.unsqueeze(grid, dim=1)
    grid_batch = grid_batch.repeat(1, len(tR1),1)

    # print(tR1.shape)
    tR1_batch = torch.unsqueeze(tR1, dim=0)
    tR1_batch = tR1_batch.repeat(num_points_square,1,1)
    tR2_batch = torch.unsqueeze(tR2, dim=0)
    tR2_batch = tR2_batch.repeat(num_points_square,1,1)
    tR3_batch = torch.unsqueeze(tR3, dim=0)
    tR3_batch = tR3_batch.repeat(num_points_square,1,1)
    # print('\n\n>>>>>>>>>num_points_square',num_points_square)
    # print('\n\n>>>>>>>>>tR1_batch',tR1_batch.shape)
    # print('\n\n>>>>>>>>>grid_batch',grid_batch.shape)
    d1_batch  = torch.norm(tR1_batch - grid_batch,dim=2)
    d2_batch  = torch.norm(tR2_batch - grid_batch,dim=2)
    d3_batch  = torch.norm(tR3_batch - grid_batch,dim=2)

    # print(tZ1.shape)
    tZ1_batch = torch.unsqueeze(tZ1, dim=0).repeat(num_points_square,1)
    tZ2_batch = torch.unsqueeze(tZ2, dim=0).repeat(num_points_square,1)
    tZ3_batch = torch.unsqueeze(tZ3, dim=0).repeat(num_points_square,1)

    zeros = torch.zeros(num_points_square, batchlen)
    zeros.to(device=torch.device('cpu'))
    ones = torch.ones(num_points_square, batchlen)
    ones.to(device=torch.device('cpu'))
    d_sumss = []
    for i in range(1,max_n+1):
        # print('\n\n>>>>>>>>>>>tZ1_batch',tZ1_batch.shape)
        # print('\n\n>>>>>>>>>>>d1_batch',d1_batch.shape)
        # print('\n\n>>>>>>>>>>>tZ2_batch',tZ2_batch.shape)
        # print('\n\n>>>>>>>>>>>d2_batch',d2_batch.shape)
        d_batch = tZ1_batch/torch.sqrt(d1_batch**2+i) + tZ2_batch/torch.sqrt(d2_batch*2+i) + tZ3_batch/torch.sqrt(d3_batch*2+i)
        if do_sum:
            d_sums = torch.sum(d_batch, dim=0)
            d_sums = torch.nn.functional.normalize(d_sums,dim=0)
        else:
            d_sums = torch.nn.functional.normalize(d_batch,dim=0)
        d_sumss.append(d_sums)
    d_sumss = torch.stack(d_sumss, dim=1)
    return d_sumss

def integration_multiple_meshes_cpu(max_n,tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum=True):
    print("diep:Performing integration")
    d_sumss = []
    grid_size = 5
        
    num_points = 2*grid_size+1
    num_points_square = num_points*num_points
    grid = [[(x, y) for x in range(-2*grid_size, 2*grid_size + 2,2)] for y in range(-2*grid_size, 2*grid_size + 2,2)]
    grid = torch.from_numpy(np.reshape(np.array(grid,dtype=np.float32), (-1,2))).to(device=torch.device('cpu'))
    d_sums = calculate_dsum_cpu(grid,tR1,tR2,tR3,num_points_square,tZ1,tZ2,tZ3,do_sum=do_sum,max_n=max_n)
    d_sumss.append(d_sums)
    if do_sum:
        d_sumss = torch.stack(d_sumss, dim=0)
        return d_sumss.reshape(len(tR1), max_n)
    else:
        d_sumss = torch.stack(d_sumss, dim=1)
        return d_sumss.squeeze()
    

def triangle_transform(R1,R2,R3,Z1,Z2,Z3):
    # import time
    # start = time.time()
    num_triplets = len(R1)
    r12 = R1-R2
    d12 = torch.sqrt((r12*r12).sum(1))
    r13 = R1-R3
    d13 = torch.sqrt((r13*r13).sum(1))
    r23 = R3 - R2
    d23 = torch.sqrt((r23*r23).sum(1))
    
    # print(R1-R2)
    # print(d12)
    # print(d12.shape)
    distances = torch.stack([d12,d13,d23],axis=1)
    # print(distances.shape)
    ordered_distances, order_indices = torch.sort(distances, dim = 1)
    # Z_array = torch.from_numpy(np.array([Z1, Z2, Z3])) # 3 x batch
    Z_array = torch.stack([Z1, Z2, Z3],dim=1) # 3 x batch

    Z_idx = torch.from_numpy(np.array([[0,1,0,2,1,2]])).repeat(num_triplets,1).to(torch.device(device))
    d1 = ordered_distances[:, 2].cpu().detach().numpy()
    d2 = ordered_distances[:, 1].cpu().detach().numpy()
    d3 = ordered_distances[:, 0].cpu().detach().numpy()

    # print(torch.stack([order_indices[:,2] * 2,order_indices[:,2] * 2+1],dim=1))
    ZZ12a = Z_idx.gather(1, torch.stack([order_indices[:,2] * 2,order_indices[:,2] * 2+1],dim=1)) # batch x 2
    ZZ13b = Z_idx.gather(1, torch.stack([order_indices[:,0] * 2,order_indices[:,0] * 2+1],dim=1))

    ZZ1arr = torch.where(ZZ12a[:,0]==ZZ13b[:,0],ZZ12a[:,0],ZZ12a[:,1])
    ones_ZZ1 = torch.ones(num_triplets)
    ZZ1arr = torch.where((ZZ12a[:,0]!=ZZ13b[:,0]) & (ZZ12a[:,1]!=ZZ13b[:,1]),ones_ZZ1,ZZ1arr)
    ZZ1 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ1arr.type(torch.int64),dim=1)).squeeze()
    ZZ2idx = torch.where(ZZ12a[:,0]==ZZ1arr,ZZ12a[:,1],ZZ12a[:,0])
    ZZ2 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ2idx.type(torch.int64),dim=1)).squeeze()
    ZZ3idx = torch.where(ZZ13b[:,0]==ZZ1arr,ZZ13b[:,1],ZZ13b[:,0])
    ZZ3 = torch.gather(Z_array, 1, torch.unsqueeze(ZZ3idx.type(torch.int64),dim=1)).squeeze()
    r1 = np.zeros([num_triplets, 2],dtype=np.float32)
    r2 = np.zeros([num_triplets, 2],dtype=np.float32)
    r2[:, 0] = d1

    r3_x = 1 / 2 * (d1 + (d3**2 - d2**2) / d1)
    r3_y = np.sqrt(d3**2 - r3_x**2)
    try:
        f = np.vectorize(lambda k: k if not np.isnan(k) else 0.0)
        r3_y = f(r3_y).astype(np.float32) # If d1 = d2 + d3, then just set r3_y to zero, it's a straightline
    except Exception as eee:
        print('r3_y',r3_y)
        print('d3',d3)
        print('r3_x',r3_x)
        print(eee)
    r3 = np.array([r3_x, r3_y]).T
    r_center = (r1 + r2 + r3) / 3

    fR1 = torch.from_numpy(r1-r_center).to(torch.device(device))
    fR2 = torch.from_numpy(r2-r_center).to(torch.device(device))
    fR3 = torch.from_numpy(r3-r_center).to(torch.device(device))
    
    return fR1,fR2,fR3,ZZ1,ZZ2,ZZ3

def integration(tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum=True):
    print("diep:Performing integration")
    grid_size = 5
    grid = [[(x, y) for x in range(-grid_size, grid_size + 1)] for y in range(-grid_size, grid_size + 1)]
    grid = torch.from_numpy(np.reshape(np.array(grid,dtype=np.float32), (-1,2)))

    batchlen = len(tR1)
    grid_batch = torch.unsqueeze(grid, dim=1)
    grid_batch = grid_batch.repeat(1, len(tR1),1).to(torch.device(device))

    # print(tR1.shape)
    tR1_batch = torch.unsqueeze(tR1, dim=0)
    tR1_batch = tR1_batch.repeat(121,1,1)
    tR2_batch = torch.unsqueeze(tR2, dim=0)
    tR2_batch = tR2_batch.repeat(121,1,1)
    tR3_batch = torch.unsqueeze(tR3, dim=0)
    tR3_batch = tR3_batch.repeat(121,1,1)

    d1_batch  = torch.norm(tR1_batch - grid_batch,dim=2)
    d2_batch  = torch.norm(tR2_batch - grid_batch,dim=2)
    d3_batch  = torch.norm(tR3_batch - grid_batch,dim=2)

    n = (
        torch.exp(-torch.norm(tR1_batch - grid_batch, dim=2) ** 2)
        + torch.exp(-torch.norm(tR2_batch - grid_batch, dim=2) ** 2)
        + torch.exp(-torch.norm(tR3_batch - grid_batch, dim=2) ** 2)
    )

    
    tZ1_batch = torch.unsqueeze(tZ1, dim=0).repeat(121,1)
    tZ2_batch = torch.unsqueeze(tZ2, dim=0).repeat(121,1)
    tZ3_batch = torch.unsqueeze(tZ3, dim=0).repeat(121,1)
    # print('tZ1',tZ1.shape)
    # print('n',n.shape)
    zeros = torch.zeros(121, batchlen)
    ones = torch.ones(121, batchlen)
    
    d_batch = n*tZ1_batch/torch.sqrt(d1_batch**2+1)
    d_batch += n*tZ2_batch/torch.sqrt(d2_batch**2+1) 
    d_batch += n*tZ3_batch/torch.sqrt(d3_batch**2+1)
    # print('d_batch',d_batch.shape)
    if do_sum:
        d_sums = torch.sum(d_batch, dim=0)
        d_sums = torch.nn.functional.normalize(d_sums,dim=0)
        return torch.reshape(d_sums, (-1,1))
    else:
        d_sums = torch.nn.functional.normalize(d_batch,dim=0)
        # print('d_sums',d_sums.shape)
        return torch.transpose(torch.squeeze(d_sums),0,1)

def calculate_dsum(grid, tR1, tR2, tR3, num_points_square, tZ1, tZ2, tZ3, do_sum=True,max_n=1):
    batchlen = len(tR1)
    grid_batch = torch.unsqueeze(grid, dim=1)
    grid_batch = grid_batch.repeat(1, len(tR1),1)

    # print(tR1.shape)
    tR1_batch = torch.unsqueeze(tR1, dim=0)
    tR1_batch = tR1_batch.repeat(num_points_square,1,1)
    tR2_batch = torch.unsqueeze(tR2, dim=0)
    tR2_batch = tR2_batch.repeat(num_points_square,1,1)
    tR3_batch = torch.unsqueeze(tR3, dim=0)
    tR3_batch = tR3_batch.repeat(num_points_square,1,1)
    # print('\n\n>>>>>>>>>num_points_square',num_points_square)
    # print('\n\n>>>>>>>>>tR1_batch',tR1_batch.shape)
    # print('\n\n>>>>>>>>>grid_batch',grid_batch.shape)
    d1_batch  = torch.norm(tR1_batch - grid_batch,dim=2)
    d2_batch  = torch.norm(tR2_batch - grid_batch,dim=2)
    d3_batch  = torch.norm(tR3_batch - grid_batch,dim=2)

    # print(tZ1.shape)
    tZ1_batch = torch.unsqueeze(tZ1, dim=0).repeat(num_points_square,1)
    tZ2_batch = torch.unsqueeze(tZ2, dim=0).repeat(num_points_square,1)
    tZ3_batch = torch.unsqueeze(tZ3, dim=0).repeat(num_points_square,1)

    zeros = torch.zeros(num_points_square, batchlen)
    ones = torch.ones(num_points_square, batchlen)
    d_sumss = []
    for i in range(1,max_n+1):
        # print('\n\n>>>>>>>>>>>tZ1_batch',tZ1_batch.shape)
        # print('\n\n>>>>>>>>>>>d1_batch',d1_batch.shape)
        # print('\n\n>>>>>>>>>>>tZ2_batch',tZ2_batch.shape)
        # print('\n\n>>>>>>>>>>>d2_batch',d2_batch.shape)
        d_batch = tZ1_batch/torch.sqrt(d1_batch**2+i) + tZ2_batch/torch.sqrt(d2_batch**2+i) + tZ3_batch/torch.sqrt(d3_batch**2+i)
        if do_sum:
            d_sums = torch.sum(d_batch, dim=0)
            d_sums = torch.nn.functional.normalize(d_sums,dim=0)
        else:
            d_sums = torch.nn.functional.normalize(d_batch,dim=0)
        d_sumss.append(d_sums)
    d_sumss = torch.stack(d_sumss, dim=1)
    return d_sumss

def integration_multiple_meshes(max_n,tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum=True):
    print("diep:Performing integration")
    d_sumss = []
    grid_size = 5
        
    num_points = 2*grid_size+1
    num_points_square = num_points*num_points
    grid = [[(x, y) for x in range(-2*grid_size, 2*grid_size + 2,2)] for y in range(-2*grid_size, 2*grid_size + 2,2)]
    grid = torch.from_numpy(np.reshape(np.array(grid,dtype=np.float32), (-1,2)))
    d_sums = calculate_dsum(grid,tR1,tR2,tR3,num_points_square,tZ1,tZ2,tZ3,do_sum=do_sum,max_n=max_n)
    d_sumss.append(d_sums)
    if do_sum:
        d_sumss = torch.stack(d_sumss, dim=0)
        return d_sumss.reshape(len(tR1), max_n)
    else:
        d_sumss = torch.stack(d_sumss, dim=1)
        return d_sumss.squeeze()
    
def three_body_dft_integral(graph,triple_bond_indices,do_sum=True):
    R1,R2,R3,Z1,Z2,Z3 = extract_vectors_from_graph(graph,triple_bond_indices)
    tR1,tR2,tR3,tZ1,tZ2,tZ3 = triangle_transform(R1,R2,R3,Z1,Z2,Z3)
    k = integration(tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum)
    return k

def three_body_dft_integral_multiple_meshes(graph,triple_bond_indices,max_n,do_sum=True):
    R1,R2,R3,Z1,Z2,Z3 = extract_vectors_from_graph(graph,triple_bond_indices)
    tR1,tR2,tR3,tZ1,tZ2,tZ3 = triangle_transform(R1,R2,R3,Z1,Z2,Z3)
    k = integration_multiple_meshes(max_n,tR1,tR2,tR3,tZ1,tZ2,tZ3,do_sum)
    return k
