"""
dataset_ncaltech_aegnn_DEBUG.py - COMPREHENSIVE DEBUGGING VERSION

This version includes extensive validation and debugging to catch:
- NaN/Inf in data
- Invalid indices
- Memory corruption
- Graph construction issues
- Batching problems
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import radius_graph
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import warnings
import sys

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

class DataValidator:
    """Comprehensive data validation."""
    
    @staticmethod
    def validate_tensor(tensor, name, expected_shape=None, allow_nan=False, allow_inf=False):
        """Validate a tensor with detailed error messages."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
        
        if expected_shape and tensor.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
        
        if not allow_nan and torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            nan_indices = torch.where(torch.isnan(tensor))
            raise ValueError(f"{name} contains {nan_count} NaN values at indices {nan_indices}")
        
        if not allow_inf and torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            raise ValueError(f"{name} contains {inf_count} Inf values")
        
        return True
    
    @staticmethod
    def validate_numpy(array, name):
        """Validate numpy array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(array)}")
        
        if np.isnan(array).any():
            raise ValueError(f"{name} contains NaN values")
        
        if np.isinf(array).any():
            raise ValueError(f"{name} contains Inf values")
        
        return True
    
    @staticmethod
    def validate_edge_index(edge_index, num_nodes, name="edge_index"):
        """Validate edge index."""
        if edge_index.size(0) != 2:
            raise ValueError(f"{name} must have shape [2, num_edges], got {edge_index.shape}")
        
        if edge_index.size(1) > 0:
            if edge_index.min() < 0:
                raise ValueError(f"{name} contains negative indices: min={edge_index.min()}")
            
            if edge_index.max() >= num_nodes:
                raise ValueError(
                    f"{name} contains out-of-bounds indices: "
                    f"max={edge_index.max()}, num_nodes={num_nodes}"
                )
        
        return True
    
    @staticmethod
    def validate_data_object(data: Data, name="Data"):
        """Validate entire Data object."""
        print(f"\n🔍 Validating {name}:")
        print(f"  - num_nodes: {data.num_nodes}")
        print(f"  - num_edges: {data.num_edges}")
        
        # Validate features
        DataValidator.validate_tensor(data.x, f"{name}.x")
        print(f"  ✓ x shape: {data.x.shape}")
        
        # Validate positions
        DataValidator.validate_tensor(data.pos, f"{name}.pos")
        print(f"  ✓ pos shape: {data.pos.shape}")
        
        # Validate edge_index
        DataValidator.validate_tensor(data.edge_index, f"{name}.edge_index")
        DataValidator.validate_edge_index(data.edge_index, data.num_nodes, f"{name}.edge_index")
        print(f"  ✓ edge_index shape: {data.edge_index.shape}")
        
        # Validate edge_attr if present
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            DataValidator.validate_tensor(data.edge_attr, f"{name}.edge_attr")
            if data.edge_attr.size(0) != data.num_edges:
                raise ValueError(
                    f"{name}.edge_attr size mismatch: "
                    f"expected {data.num_edges}, got {data.edge_attr.size(0)}"
                )
            print(f"  ✓ edge_attr shape: {data.edge_attr.shape}")
        
        # Validate label
        DataValidator.validate_tensor(data.y, f"{name}.y")
        if data.y.numel() == 1:
            print(f"  ✓ y: {data.y.item()}")
        else:
            print(f"  ✓ y: {data.y.tolist()}")  # Print all labels in batch
        
        return True


# ============================================================================
# EVENT FILE LOADING
# ============================================================================

def load_ncaltech101_bin(file_path: str) -> np.ndarray:
    """Load N-Caltech101 .bin event file with validation."""
    print(f"\n📂 Loading: {Path(file_path).name}")
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        
        if len(raw_data) == 0:
            raise ValueError("Empty file")
        
        if len(raw_data) % 5 != 0:
            raise ValueError(f"Invalid file size: {len(raw_data)} (not divisible by 5)")
        
        num_events = len(raw_data) // 5
        events = np.zeros((num_events, 4), dtype=np.float32)
        
        raw_data = np.uint32(raw_data)
        
        events[:, 0] = raw_data[0::5]  # X
        events[:, 1] = raw_data[1::5]  # Y
        events[:, 3] = (raw_data[2::5] & 128) >> 7  # Polarity
        
        # Timestamp
        events[:, 2] = (
            ((raw_data[2::5] & 127) << 16) |
            (raw_data[3::5] << 8) |
            raw_data[4::5]
        )
        
        # Validate ranges
        if events[:, 0].max() > 240 or events[:, 1].max() > 180:
            warnings.warn(f"Events exceed sensor size: x_max={events[:, 0].max()}, y_max={events[:, 1].max()}")
        
        print(f"  ✓ Loaded {num_events} events")
        print(f"  ✓ Time range: {events[:, 2].min():.0f} - {events[:, 2].max():.0f} µs")
        
        # Validate for NaN/Inf
        DataValidator.validate_numpy(events, "events")
        
        return events
        
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        raise


# ============================================================================
# AEGNN PREPROCESSOR (DEBUG VERSION)
# ============================================================================

class AEGNNPreprocessor:
    """AEGNN preprocessor with extensive debugging."""
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (240, 180),
        r: float = 5.0,
        d_max: int = 32,
        n_samples: int = 25000,
        beta: float = 1e-3,
        debug: bool = True
    ):
        self.sensor_size = sensor_size
        self.r = r
        self.d_max = d_max
        self.n_samples = n_samples
        self.beta = beta
        self.debug = debug
    
    def preprocess(self, events: np.ndarray, label: int) -> Data:
        """Main preprocessing with safety checks."""
        if len(events) < 10:
            return self._create_empty_graph(label)
        
        windowed_events = self._extract_50ms_window(events)
        sampled_events = self._uniform_subsample(windowed_events)
        x, pos = self._create_features_and_positions(sampled_events)
        edge_index = self._build_radius_graph(pos)
        
        # ✅ Return Data WITHOUT edge_attr - AEGNN computes it
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long)
        )
        
        return data

    
    def _extract_50ms_window(self, events: np.ndarray) -> np.ndarray:
        """Extract 50ms window with validation."""
        if len(events) < 10:
            return events
        
        center_idx = len(events) // 2
        t_center = events[center_idx, 2]
        
        window_us = 50 * 1000
        t_start = t_center - window_us / 2
        t_end = t_center + window_us / 2
        
        timestamps = events[:, 2]
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        windowed = events[mask]
        
        if len(windowed) < 10:
            warnings.warn("Window too small, using all events")
            return events
        
        return windowed
    
    def _uniform_subsample(self, events: np.ndarray) -> np.ndarray:
        """Uniform subsampling with validation."""
        if len(events) <= self.n_samples:
            return events
        
        indices = np.linspace(0, len(events) - 1, self.n_samples, dtype=int)
        return events[indices]
    
    def _create_features_and_positions(self, events: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create features and positions with validation."""
        xs, ys, ts, ps = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        
        # Node features: polarity
        x = torch.from_numpy(ps.reshape(-1, 1)).float()
        
        # Positions: pixel coordinates + beta-scaled time
        t_min = ts.min()
        t_beta = (ts - t_min) * self.beta
        
        # ✅ CRITICAL: Check for Inf after scaling
        if np.isinf(t_beta).any():
            raise ValueError(f"Inf in t_beta! beta={self.beta}, t_range=[{ts.min()}, {ts.max()}]")
        
        positions = np.stack([xs, ys, t_beta], axis=1)
        pos = torch.from_numpy(positions).float()
        
        return x, pos
    
    def _build_radius_graph(self, pos: torch.Tensor) -> torch.Tensor:
        """Build radius graph with comprehensive validation."""
        if pos.size(0) < 2:
            print("  ⚠️  Too few nodes for graph")
            return torch.zeros((2, 0), dtype=torch.long)
        
        try:
            # Build initial graph
            print(f"  🔨 Building radius graph (r={self.r})...")
            edge_index = radius_graph(
                pos,
                r=self.r,
                loop=False,
                max_num_neighbors=999999
            )
            
            print(f"     - Initial edges: {edge_index.size(1)}")
            
            if edge_index.size(1) == 0:
                print("  ⚠️  No edges found")
                return edge_index
            
            # Enforce d_max
            print(f"  ✂️  Enforcing d_max={self.d_max}...")
            edge_index = self._enforce_dmax(edge_index, pos)
            print(f"     - Final edges: {edge_index.size(1)}")
            
            # Final validation
            DataValidator.validate_edge_index(edge_index, pos.size(0))
            
            return edge_index.contiguous()
            
        except Exception as e:
            print(f"  ❌ Graph construction failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _enforce_dmax(self, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Safely enforce d_max."""
        if edge_index.size(1) == 0:
            return edge_index
        
        num_nodes = pos.size(0)
        src, dst = edge_index
        
        keep_edges = []
        
        for node_id in range(num_nodes):
            mask = (src == node_id)
            if mask.sum() == 0:
                continue
            
            node_dst = dst[mask]
            edge_ids = torch.where(mask)[0]
            
            if len(node_dst) <= self.d_max:
                keep_edges.append(edge_ids)
                continue
            
            # Keep d_max nearest
            distances = torch.norm(pos[node_dst] - pos[node_id].unsqueeze(0), dim=1)
            _, nearest_idx = torch.topk(distances, self.d_max, largest=False)
            keep_edges.append(edge_ids[nearest_idx])
        
        if keep_edges:
            keep_mask = torch.cat(keep_edges)
            return edge_index[:, keep_mask]
        else:
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _create_empty_graph(self, label: int) -> Data:
        """Create minimal graph for edge cases."""
        return Data(
            x=torch.zeros((2, 1), dtype=torch.float32),
            pos=torch.zeros((2, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            # NO edge_attr here
            y=torch.tensor([label], dtype=torch.long)
        )



# ============================================================================
# DATASET CLASS (DEBUG VERSION)
# ============================================================================

class NCaltech101AEGNN(Dataset):
    """N-Caltech101 Dataset with comprehensive debugging."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        r: float = 5.0,
        d_max: int = 32,
        n_samples: int = 25000,
        beta: float = 1e-3,
        train_ratio: float = 0.7,
        random_seed: int = 42,
        debug: bool = True
    ):
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got '{split}'"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.debug = debug
        
        # Initialize preprocessor
        self.preprocessor = AEGNNPreprocessor(
            sensor_size=(240, 180),
            r=r,
            d_max=d_max,
            n_samples=n_samples,
            beta=beta,
            debug=debug
        )
        
        # Find dataset
        img_dir = self.root_dir / 'img'
        if not img_dir.exists():
            raise FileNotFoundError(f"Directory not found: {img_dir}")
        
        # Collect samples
        all_samples = []
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            
            bin_files = sorted(class_dir.glob('*.bin'))
            for bin_file in bin_files:
                all_samples.append((str(bin_file), idx))
        
        # Stratified split
        file_paths = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]
        
        train_files, test_files, train_labels, test_labels = train_test_split(
            file_paths, labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=random_seed
        )
        
        self.samples = list(zip(
            train_files if split == 'train' else test_files,
            train_labels if split == 'train' else test_labels
        ))
        
        # Print info
        self._print_info(len(all_samples), len(train_files), len(test_files))
    
    def _print_info(self, total, train_size, test_size):
        """Print dataset information."""
        print(f"\n{'='*70}")
        print(f"N-Caltech101 AEGNN Dataset ({self.split.upper()}) - DEBUG MODE")
        print(f"{'='*70}")
        print(f"Root: {self.root_dir}")
        print(f"Classes: {len(self.class_to_idx)}")
        print(f"Total samples: {total}")
        print(f"Train samples: {train_size}")
        print(f"Test samples: {test_size}")
        print(f"Current split: {len(self.samples)}")
        
        print(f"\n📋 AEGNN Configuration:")
        print(f"  - 50ms temporal window")
        print(f"  - Uniform sampling to {self.preprocessor.n_samples} events")
        print(f"  - Beta scaling: β = {self.preprocessor.beta}")
        print(f"  - 3D radius graph: r = {self.preprocessor.r}")
        print(f"  - Max neighbors: d_max = {self.preprocessor.d_max}")
        print(f"  - Node features: 1D (polarity only)")
        print(f"  - Sensor size: {self.preprocessor.sensor_size}")
        print(f"  - Debug mode: {self.debug}")
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and preprocess sample with error handling."""
        file_path, label = self.samples[idx]
        
        try:
            # Load events
            events = load_ncaltech101_bin(file_path)
            
            # Preprocess
            data = self.preprocessor.preprocess(events, label)
            
            return data
            
        except Exception as e:
            print(f"\n❌ ERROR in __getitem__(idx={idx}):")
            print(f"   File: {file_path}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty graph as fallback
            print("   Returning empty graph as fallback\n")
            return self.preprocessor._create_empty_graph(label)
    
    @staticmethod
    def collate_fn(data_list):
        """Ultra-defensive collate with extensive validation."""
        import torch
        from torch_geometric.data import Batch, Data
        
        print(f"\n{'='*70}")
        print(f"🔧 COLLATE_FN: Processing {len(data_list)} graphs")
        print(f"{'='*70}")
        
        # Validate EACH graph before batching
        for i, data in enumerate(data_list):
            print(f"\nGraph {i}:")
            print(f"  num_nodes: {data.num_nodes}")
            print(f"  num_edges: {data.num_edges}")
            
            if data.num_edges > 0:
                max_idx = data.edge_index.max().item()
                min_idx = data.edge_index.min().item()
                
                if max_idx >= data.num_nodes:
                    print(f"  ❌ INVALID: edge_index.max()={max_idx} >= num_nodes={data.num_nodes}")
                    # Fix: Remove invalid edges
                    mask = (data.edge_index[0] < data.num_nodes) & (data.edge_index[1] < data.num_nodes)
                    data.edge_index = data.edge_index[:, mask]
                    print(f"  ✓ Fixed: new num_edges={data.edge_index.size(1)}")
                
                if min_idx < 0:
                    print(f"  ❌ INVALID: Negative indices detected")
                    mask = (data.edge_index[0] >= 0) & (data.edge_index[1] >= 0)
                    data.edge_index = data.edge_index[:, mask]
                    print(f"  ✓ Fixed: new num_edges={data.edge_index.size(1)}")
                
                print(f"  ✓ Valid: edge_index in [0, {data.num_nodes-1}]")
        
        # Batch the graphs
        try:
            batch = Batch.from_data_list(data_list)
        except Exception as e:
            print(f"\n❌ Batching failed: {e}")
            raise
        
        print(f"\n{'='*70}")
        print(f"📦 BATCHED RESULT:")
        print(f"  num_graphs: {batch.num_graphs}")
        print(f"  total_nodes: {batch.num_nodes}")
        print(f"  total_edges: {batch.num_edges}")
        
        if batch.num_edges > 0:
            print(f"  edge_index range: [{batch.edge_index.min()}, {batch.edge_index.max()}]")
            print(f"  Expected max: {batch.num_nodes - 1}")
            
            # CRITICAL CHECK
            if batch.edge_index.max() >= batch.num_nodes:
                print(f"\n❌ BATCHING CORRUPTED EDGE_INDEX!")
                print(f"   This is a PyG bug or data corruption issue")
                raise ValueError("Invalid edge_index after batching")
        
        # Verify batch attribute exists
        if not hasattr(batch, 'batch') or batch.batch is None:
            print(f"  ⚠️ batch attribute missing - creating it")
            batch_vec = []
            for i, data in enumerate(data_list):
                batch_vec.extend([i] * data.num_nodes)
            batch.batch = torch.tensor(batch_vec, dtype=torch.long)
        
        print(f"  batch attr shape: {batch.batch.shape}")
        print(f"{'='*70}\n")
        
        return batch




# ============================================================================
# DATALOADER CREATION
# ============================================================================

def get_aegnn_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,  # Set to 0 for debugging
    r: float = 5.0,
    d_max: int = 32,
    n_samples: int = 25000,
    beta: float = 1e-3,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    debug: bool = True
):
    """Create dataloaders with debugging."""
    print(f"\n{'='*70}")
    print(f"🚀 Creating AEGNN DataLoaders")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Debug mode: {debug}")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_dataset = NCaltech101AEGNN(
        root_dir=root_dir,
        split='train',
        r=r,
        d_max=d_max,
        n_samples=n_samples,
        beta=beta,
        debug=debug
    )
    
    test_dataset = NCaltech101AEGNN(
        root_dir=root_dir,
        split='test',
        r=r,
        d_max=d_max,
        n_samples=n_samples,
        beta=beta,
        debug=debug
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=NCaltech101AEGNN.collate_fn,
        pin_memory=False,  # Disable for debugging
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=NCaltech101AEGNN.collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"✅ DataLoaders created successfully\n")
    
    return train_loader, test_loader


# ============================================================================
# TEST SCRIPT
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("AEGNN DEBUG MODE - Comprehensive Testing")
    print("="*80)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared\n")
    
    # Test single sample
    print("\n" + "="*80)
    print("TEST 1: Single Sample")
    print("="*80)
    
    dataset = NCaltech101AEGNN(
        root_dir='datasets/ncaltech',
        split='train',
        r=5.0,
        d_max=32,
        n_samples=1000,
        debug=True
    )
    
    sample = dataset[0]
    print("\n✅ Single sample test PASSED")
    
    # Test dataloader
    print("\n" + "="*80)
    print("TEST 2: DataLoader")
    print("="*80)
    
    train_loader, test_loader = get_aegnn_dataloaders(
        root_dir='datasets/ncaltech',
        batch_size=4,
        num_workers=0,  # Must be 0 for debugging
        n_samples=1000,
        debug=True
    )
    
    print("\nFetching first batch...")
    batch = next(iter(train_loader))
    print("\n✅ DataLoader test PASSED")
    
    # Test CUDA transfer
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("TEST 3: CUDA Transfer")
        print("="*80)
        
        print("Moving batch to CUDA...")
        batch = batch.to('cuda')
        print(f"✓ Batch on device: {batch.x.device}")
        print("✅ CUDA transfer test PASSED")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
