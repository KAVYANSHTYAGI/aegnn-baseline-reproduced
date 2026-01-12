import torch
import torch_geometric
import pytorch_lightning as pl
import torchmetrics.functional as pl_metrics


from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name


class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, **model_kwargs):
        super(RecognitionModel, self).__init__()
        self.num_outputs = num_classes
        self.dim = dim

        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_classes, **model_kwargs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        """Forward pass with robust edge_attr handling."""
        # Truncate positions to required dimensions
        data.pos = data.pos[:, :self.dim]
        
        # ✅ CRITICAL: Validate positions BEFORE computing edge_attr
        if torch.isnan(data.pos).any():
            print(f"⚠️ NaN detected in pos!")
            data.pos = torch.nan_to_num(data.pos, nan=0.0)
        
        if torch.isinf(data.pos).any():
            print(f"⚠️ Inf detected in pos!")
            data.pos = torch.nan_to_num(data.pos, posinf=1000.0, neginf=-1000.0)
        
        # Compute edge_attr from positions
        if getattr(data, "edge_attr", None) is None and data.num_edges > 0:
            src, dst = data.edge_index
            # Compute relative positions
            edge_attr = data.pos[dst] - data.pos[src]
            
            # ✅ CRITICAL: Validate and clamp edge_attr
            if torch.isnan(edge_attr).any():
                print(f"⚠️ NaN in computed edge_attr!")
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
            
            if torch.isinf(edge_attr).any():
                print(f"⚠️ Inf in computed edge_attr!")
                edge_attr = torch.nan_to_num(edge_attr, posinf=100.0, neginf=-100.0)
            
            # Clamp to reasonable range
            edge_attr = torch.clamp(edge_attr, min=-1000.0, max=1000.0)
            
            data.edge_attr = edge_attr[:, :self.dim]
        elif hasattr(data, "edge_attr") and data.edge_attr is not None:
            # Validate existing edge_attr
            data.edge_attr = data.edge_attr[:, :self.dim]
            
            if torch.isnan(data.edge_attr).any():
                print(f"⚠️ NaN in provided edge_attr!")
                data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
            
            if torch.isinf(data.edge_attr).any():
                print(f"⚠️ Inf in provided edge_attr!")
                data.edge_attr = torch.nan_to_num(data.edge_attr, posinf=100.0, neginf=-100.0)
        
        return self.model.forward(data)


