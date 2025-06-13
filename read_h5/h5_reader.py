import os
import h5py
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import psutil
from tqdm import tqdm
# ------------------------------------------------------------------------------------- 
# UNet h5 Reader.
#
# Reads .h5 files containing image and mask datasets, trains a U-Net model on the data, 
# and saves the trained model.
#
# Usage:
#   python h5_reader.py
# -------------------------------------------------------------------------------------
class UNet(nn.Module):
    """A simple U-Net implementation for image segmentation."""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # Encoder.
        self.enc1 = self.conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck.
        self.bottleneck = self.conv_block(32, 64)
        # Decoder.
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = self.conv_block(64 + 32, 32)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def conv_block(self, in_channels, out_channels):
        """Helper function to create a block of two convolutional layers with 
        ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True))
    def forward(self, x):
        """Forward pass of the U-Net."""
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        b = self.bottleneck(p2)
        u3 = self.up3(b)
        u3 = torch.cat([u3, c2], dim=1)
        c3 = self.dec3(u3)
        u4 = self.up4(c3)
        u4 = torch.cat([u4, c1], dim=1)
        c4 = self.dec4(u4)
        out = self.final(c4)
        return self.sigmoid(out)

class H5Dataset(Dataset):
    """PyTorch Dataset for loading images and masks from numpy arrays."""
    def __init__(self, X, Y):
        """Constructor function for H5Dataset."""
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2)
        self.Y = torch.from_numpy(Y).permute(0, 3, 1, 2)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataset(f, key, filepath):
    """Helper to retrieve a dataset from an HDF5 file, handling both 
    datasets and groups."""
    if isinstance(f[key], h5py.Dataset):
        return f[key][:]
    elif isinstance(f[key], h5py.Group):
        # If it's a group, return the first dataset found in the group
        for subkey in f[key].keys():
            if isinstance(f[key][subkey], h5py.Dataset):
                return f[key][subkey][:]
        raise ValueError(f"No dataset found in group '{key}' in {filepath}")
    else:
        raise ValueError(f"Unknown type for '{key}' in {filepath}")

def load_h5_data(filepath):
    """Loads image and mask data from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        X = get_dataset(f, 'images', filepath)
        Y = get_dataset(f, 'masks', filepath)
    # Ensure data has a channel dimension
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if Y.ndim == 3:
        Y = Y[..., np.newaxis]
    return X.astype(np.float32), Y.astype(np.float32)

def train_on_file(filepath, 
    output_dir='trained_models', 
    device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Loads data, trains a U-Net model, evaluates, and saves the model 
    for a single .h5 file. Returns training and testing statistics."""
    try:
        X_train, Y_train = load_h5_data(filepath)
        dataset = H5Dataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = UNet(in_channels=X_train.shape[3], out_channels=1).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        epoch_losses = []
        num_epochs = 5
        # Progress bar for epochs (hidden in main progress bar).
        for epoch in tqdm(range(num_epochs), desc=f"Training {os.path.basename(filepath)}", leave=False):
            epoch_loss = 0
            for X_batch, Y_batch in dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
        # Save model.
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(filepath).replace('.h5', '_unet_model.pt')
        torch.save(model.state_dict(), os.path.join(output_dir, model_name))
        # Evaluate on training data (as a proxy for testing).
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2).to(device)
            Y_tensor = torch.from_numpy(Y_train).permute(0, 3, 1, 2).to(device)
            outputs = model(X_tensor)
            test_loss = criterion(outputs, Y_tensor).item()
            # Simple accuracy: threshold at 0.5.
            preds = (outputs > 0.5).float()
            accuracy = (preds == Y_tensor).float().mean().item()
        # Return stats for summary.
        return {
            "file": os.path.basename(filepath),
            "final_train_loss": epoch_losses[-1],
            "test_loss": test_loss,
            "test_accuracy": accuracy}
    except Exception as e:
        # Return error info for summary.
        return {
            "file": os.path.basename(filepath),
            "error": str(e)}

def process_all_h5(directory='imorphics-cartilage', ram_per_worker_gb=2):
    """Batch Processing. Processes all .h5 files in the given directory using 
    multiprocessing. Returns a list of results for each file."""
    h5_files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.h5')]
    if not h5_files:
        return []
    # Determine resources for parallel processing.
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_cores = os.cpu_count() or 1
    max_workers_ram = max(1, int(available_gb // ram_per_worker_gb))
    max_workers = 1  # Set to 1 for stability, increase for parallelism if resources allow.
    results = []
    # Progress bar for all files.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_on_file, file): file for file in h5_files}
        for future in tqdm(concurrent.futures.as_completed(futures), 
                        total=len(h5_files), desc="Overall Progress"):
            result = future.result()
            results.append(result)
    return results

def read_h5_to_tensor(file_path, dataset_name):
    """Reads a dataset from an HDF5 file and returns it as a PyTorch tensor."""
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
        tensor = torch.tensor(data)
    return tensor

def display_h5_contents(filepath):
    """Prints the structure and contents of an HDF5 file, including groups and datasets."""
    with h5py.File(filepath, 'r') as f:
        print(f"Contents of {filepath}:")
        for key in f.keys():
            print(f"  {key}: {type(f[key])}")
            if isinstance(f[key], h5py.Group):
                print(f"    Subkeys: {list(f[key].keys())}")
                for subkey in f[key].keys():
                    item = f[key][subkey]
                    if isinstance(item, h5py.Dataset):
                        print(f"      {subkey}: Dataset, shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"      {subkey}: Group")
            elif isinstance(f[key], h5py.Dataset):
                print(f"    Dataset, shape={f[key].shape}, dtype={f[key].dtype}")

def main():
    """Main driver function. Trains models on all .h5 files and prints summary statistics."""
    directory = 'imorphics-cartilage'
    results = process_all_h5(directory)
    # Print summary stats after all training is done.
    print("\nFinal Training and Testing Stats:")
    for res in results:
        if "error" in res:
            print(f"{res['file']}: ERROR - {res['error']}")
        else:
            print(f"{res['file']}: Final Train Loss={res['final_train_loss']:.4f}, "
                  f"Test Loss={res['test_loss']:.4f}, Test Accuracy={res['test_accuracy']:.4f}")

# The big red activation button.
if __name__ == "__main__":
    main()
