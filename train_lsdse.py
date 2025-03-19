import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# CUDA availability check
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("Using CPU for training")

# Dataset paths and functions
# UPDATED PATH to dataset
dataset_path = "datasets/LSDSE-Dataset/data"

# UPDATED PATH for cross sections
with open("datasets/LSDSE-Dataset/cross_sections.json", "r") as f:
    cross_sections = json.load(f)
    print(f"Loaded {len(cross_sections)} cross-section types")

# List all JSON files in dataset - UPDATED PATTERN to match data0.json to data39.json
json_files = glob(os.path.join(dataset_path, "data*.json"))
print(f"Found {len(json_files)} building structure files")

# Function to convert LSDSE graph structure to PyTorch Geometric format
def convert_to_pytorch_geometric(building_data):
    """
    Convert an LSDSE building structure to PyTorch Geometric format.
    """
    # Extract node features
    # Format: [x1, y1, z1, x2, y2, z2, cross_section_one_hot, if_beam, if_boundary_beam, if_roof_beam, metal_deck_area]
    
    # Get bar node locations (endpoints)
    bar_locations = np.array(building_data["bar_nodes"]["end_point_location"])
    
    # Get one-hot cross section features
    cross_sections_features = np.array(building_data["bar_nodes"]["cross_section"])
    
    # Get other beam properties
    if_beam = np.array(building_data["bar_nodes"]["if_beam"]).reshape(-1, 1)
    if_boundary_beam = np.array(building_data["bar_nodes"]["if_boundary_beam"]).reshape(-1, 1)
    if_roof_beam = np.array(building_data["bar_nodes"]["if_roof_beam"]).reshape(-1, 1)
    metal_deck_area = np.array(building_data["bar_nodes"]["metal_deck_area_on_beam"]).reshape(-1, 1)
    
    # Combine all node features
    node_features = np.hstack((
        bar_locations,         # 6 features (x1,y1,z1,x2,y2,z2)
        cross_sections_features, # 9 features (one-hot cross section)
        if_beam,               # 1 feature
        if_boundary_beam,      # 1 feature
        if_roof_beam,          # 1 feature
        metal_deck_area        # 1 feature
    ))
    
    # Create edge index tensor from senders and receivers
    edge_list = []
    
    # Column-column edges
    if len(building_data["edges"]["column_column_senders"]) > 0:
        col_col_edges = np.stack([
            building_data["edges"]["column_column_senders"],
            building_data["edges"]["column_column_receivers"]
        ], axis=0)
        edge_list.append(col_col_edges)
    
    # Beam-column edges
    if len(building_data["edges"]["beam_column_senders"]) > 0:
        beam_col_edges = np.stack([
            building_data["edges"]["beam_column_senders"],
            building_data["edges"]["beam_column_receivers"]
        ], axis=0)
        edge_list.append(beam_col_edges)
    
    # Column-ground edges
    if len(building_data["edges"]["column_grounds_senders"]) > 0:
        col_ground_edges = np.stack([
            building_data["edges"]["column_grounds_senders"],
            building_data["edges"]["column_grounds_receivers"]
        ], axis=0)
        edge_list.append(col_ground_edges)
    
    # Combine all edge types
    edge_index = np.concatenate(edge_list, axis=1) if edge_list else np.empty((2, 0), dtype=np.int64)
    
    # Extract drift ratios (target values for ML task)
    drift_ratios = np.array(building_data["drift_ratio"][:-1])  # Exclude ground node
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(drift_ratios, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(node_features)
    )
    
    return data

# Memory-efficient dataset class
class LSDSEDataset(Dataset):
    def __init__(self, json_files, transform=None, pre_transform=None):
        super(LSDSEDataset, self).__init__(None, transform, pre_transform)
        self.json_files = json_files
        
        # Count total graphs (this reads all files once)
        self.total_graphs = 0
        for json_file in json_files:
            with open(json_file, "r") as f:
                self.total_graphs += len(json.load(f))
        
        # Create a mapping from index to (file_idx, structure_idx)
        self.index_map = []
        current_idx = 0
        for file_idx, json_file in enumerate(json_files):
            with open(json_file, "r") as f:
                num_structures = len(json.load(f))
                for struct_idx in range(num_structures):
                    self.index_map.append((file_idx, struct_idx))
                    current_idx += 1
    
    def len(self):
        return self.total_graphs
    
    def get(self, idx):
        file_idx, struct_idx = self.index_map[idx]
        json_file = self.json_files[file_idx]
        
        with open(json_file, "r") as f:
            building_structures = json.load(f)
            building = building_structures[struct_idx]
            return convert_to_pytorch_geometric(building)

def visualize_examples(dataset, num_examples=3):
    # Visualize a few examples
    for i in range(min(num_examples, len(dataset))):
        data = dataset[i]
        
        # Print information about the graph
        print(f"Graph {i}:")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.edge_index.size(1)}")
        print(f"  - Features: {data.x.size(1)}")
        print(f"  - Target shape: {data.y.shape}")
        
        # Optional: Plot a simple visualization 
        # This assumes bar_nodes are arranged as [x1,y1,z1,x2,y2,z2,...] and uses just x,z coordinates
        plt.figure(figsize=(10, 8))
        
        # Extract endpoints from features (first 6 columns)
        for j in range(data.num_nodes):
            x1, y1, z1, x2, y2, z2 = data.x[j, 0:6].cpu().numpy()
            
            # Use different colors for beams and columns (feature at index 15 is if_beam)
            color = 'red' if data.x[j, 15] > 0.5 else 'blue'
            
            plt.plot([x1, x2], [z1, z2], color=color, linewidth=2)
        
        plt.title(f"Building Structure - Example {i}")
        plt.xlabel("X-axis (ft)")
        plt.ylabel("Z-axis (ft)")
        plt.grid(True)
        plt.show()

# Main execution function
def main():
    # REDUCED number of files to process initially for testing
    max_files_to_process = 2  # Start small, increase as you verify memory usage
    
    # Create the dataset
    dataset = LSDSEDataset(json_files[:max_files_to_process])
    print(f"Dataset size: {len(dataset)} structures")
    
    # Visualize examples - uncomment if you need visualization
    # visualize_examples(dataset)
    
    # Create train/val/test split
    # 70/15/15 split
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
    
    # Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders with appropriate batch size
    batch_size = 8  # REDUCED batch size for better memory management
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define GraphSAGE model
    class StructuralGraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
            super(StructuralGraphSAGE, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            
            # First layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
            # Output layer
            self.convs.append(SAGEConv(hidden_channels, out_channels))
            
        def forward(self, x, edge_index):
            # Graph representation learning
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
            
            # Final layer
            x = self.convs[-1](x, edge_index)
            return x
    
    # Get input dimension from data
    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)  # Should be 19
    hidden_channels = 64
    out_channels = 4  # [Ex_x, Ex_y, Ey_x, Ey_y] drift ratios
    
    # Initialize model and move to device
    model = StructuralGraphSAGE(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Training function with CUDA support
    def train():
        model.train()
        total_loss = 0
        
        for data in train_loader:
            # Move data to device
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Target: drift ratios for each story
            target = data.y.view(-1, out_channels)
            
            # Calculate loss
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
        
        return total_loss / len(train_loader.dataset)
    
    # Evaluation function with CUDA support
    def evaluate(loader):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data in loader:
                # Move data to device
                data = data.to(device)
                out = model(data.x, data.edge_index)
                target = data.y.view(-1, out_channels)
                loss = criterion(out, target)
                total_loss += loss.item() * data.num_graphs
        
        return total_loss / len(loader.dataset)
    
    # Train the model
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # ADDED checkpointing for longer training runs
    checkpoint_interval = 10  # Save every 10 epochs
    
    for epoch in range(num_epochs):
        train_loss = train()
        val_loss = evaluate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_structural_model.pt")
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluate on test set
    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # Make predictions using trained model
    # Load best model
    model.load_state_dict(torch.load("best_structural_model.pt"))
    model.eval()
    
    # Predict on a few test samples
    sample_idx = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i > 0:  # Just look at first batch
                break
                
            data = data.to(device)
            predictions = model(data.x, data.edge_index)
            targets = data.y.view(-1, out_channels)
            
            # Convert back to numpy for easier analysis
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Show predictions vs targets for first few samples
            for j in range(min(5, len(predictions))):
                print(f"Sample {sample_idx + j}:")
                print(f"  Predicted drift ratios: {predictions[j]}")
                print(f"  Actual drift ratios: {targets[j]}")
                print(f"  Difference: {np.abs(predictions[j] - targets[j])}")
                print()

if __name__ == "__main__":
    main() 