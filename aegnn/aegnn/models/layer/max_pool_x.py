import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
from typing import List, Optional


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        # Store as list, will convert to tensor on correct device in forward
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None):
        # ✅ FIX: Convert voxel_size to tensor on same device as pos
        voxel_size_tensor = torch.tensor(self.voxel_size, dtype=torch.float, device=pos.device)
        
        # Create voxel clusters
        cluster = voxel_grid(pos, batch=batch, size=voxel_size_tensor)
        
        # ✅ FIX: Pool nodes by cluster using scatter
        num_clusters = int(cluster.max().item()) + 1
        pooled_x = scatter(x, cluster, dim=0, dim_size=num_clusters, reduce='max')
        
        # ✅ Determine which batch each cluster belongs to
        cluster_batch = scatter(batch, cluster, dim=0, dim_size=num_clusters, reduce='min')
        
        # ✅ Create output with exactly (batch_size * self.size) nodes
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
            num_features = pooled_x.size(1)
            
            out = torch.zeros(batch_size * self.size, num_features, 
                            dtype=pooled_x.dtype, device=pooled_x.device)
            
            # Assign clusters to output positions
            for b in range(batch_size):
                # Get all clusters belonging to this batch
                mask = cluster_batch == b
                batch_clusters = pooled_x[mask]
                
                # Take up to self.size clusters, pad if fewer
                n = min(batch_clusters.size(0), self.size)
                if n > 0:
                    out[b * self.size : b * self.size + n] = batch_clusters[:n]
            
            return out
        else:
            return pooled_x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"
