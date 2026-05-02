"""Data loading utilities for ModelNet10/40."""

import os
import glob
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import requests
from tqdm import tqdm


def download_modelnet(data_dir='./data', dataset='modelnet10'):

    os.makedirs(data_dir, exist_ok=True)

    if dataset.lower() == 'modelnet10':
        url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
        zip_path = os.path.join(data_dir, 'ModelNet10.zip')
        extract_path = os.path.join(data_dir, 'ModelNet10')
    elif dataset.lower() == 'modelnet40':
        url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
        zip_path = os.path.join(data_dir, 'ModelNet40.zip')
        extract_path = os.path.join(data_dir, 'ModelNet40')
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'modelnet10' or 'modelnet40'")

    if os.path.exists(extract_path):
        print(f"✓ {dataset} already exists at {extract_path}")
        return extract_path

    print(f"Downloading {dataset} from {url}...")
    print("This may take several minutes...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)
    print(f"✓ {dataset} downloaded and extracted to {extract_path}")
    return extract_path



def read_off_file(file_path):
    """
    Read OFF file and return vertices and faces.

    Args:
        file_path: path to .off file

    Returns:
        vertices: (N, 3) numpy array
        faces: (M, 3) numpy array
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove empty lines and comments
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

    # Check if first line is 'OFF'
    if lines[0] == 'OFF':
        lines = lines[1:]
    elif lines[0].startswith('OFF'):
        # Sometimes 'OFF' and counts are on the same line
        lines[0] = lines[0][3:].strip()

    # Read header
    n_vertices, n_faces, _ = map(int, lines[0].split())

    # Read vertices
    vertices = []
    for i in range(1, n_vertices + 1):
        vertex = list(map(float, lines[i].split()[:3]))
        vertices.append(vertex)
    vertices = np.array(vertices, dtype=np.float32)

    # Read faces
    faces = []
    for i in range(n_vertices + 1, n_vertices + n_faces + 1):
        if i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 4:
                face = list(map(int, parts[1:4]))
                faces.append(face)
    faces = np.array(faces, dtype=np.int32) if faces else np.array([], dtype=np.int32)

    return vertices, faces

def sample_point_cloud(vertices, faces, n_points=1024):
    """
    Sample point cloud from mesh.

    Args:
        vertices: (N, 3) mesh vertices
        faces: (M, 3) mesh faces
        n_points: number of points to sample

    Returns:
        point_cloud: (n_points, 3) sampled points
    """
    if len(faces) == 0:
        # No faces - just sample from vertices with repetition
        if len(vertices) >= n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_points, replace=True)
        point_cloud = vertices[indices]
    else:
        try:
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # Sample points from surface
            point_cloud, _ = trimesh.sample.sample_surface(mesh, n_points)
        except:
            # Fallback: sample from vertices
            if len(vertices) >= n_points:
                indices = np.random.choice(len(vertices), n_points, replace=False)
            else:
                indices = np.random.choice(len(vertices), n_points, replace=True)
            point_cloud = vertices[indices]

    # Normalize point cloud
    point_cloud = point_cloud - point_cloud.mean(axis=0)
    point_cloud = point_cloud / np.abs(point_cloud).max()

    return point_cloud.astype(np.float32)


class ModelNetDataset(Dataset):

    def __init__(self, root_dir, split='train', n_points=1024,
                 data_augmentation=True, dataset='modelnet10'):

        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.data_augmentation = data_augmentation and (split == 'train')

        # Get class names
        self.classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all files
        self.files = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name, split)
            if not os.path.exists(class_dir):
                continue

            class_files = glob.glob(os.path.join(class_dir, '*.off'))
            for file_path in class_files:
                self.files.append(file_path)
                self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.files)} {split} samples from {dataset}")
        print(f"Classes ({len(self.classes)}): {', '.join(self.classes)}")

    def __len__(self):
        return len(self.files)

    def random_rotation_matrix(self):
        angles = np.random.uniform(0, 2*np.pi, 3)

        # Rotation around z-axis (most common for upright objects)
        theta = angles[2]
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        return R_z.astype(np.float32)

    def __getitem__(self, idx):
        # Load mesh
        file_path = self.files[idx]
        label = self.labels[idx]

        vertices, faces = read_off_file(file_path)
        point_cloud = sample_point_cloud(vertices, faces, self.n_points)

        # Data augmentation
        if self.data_augmentation:
            # Random rotation
            R = self.random_rotation_matrix()
            point_cloud = (R @ point_cloud.T).T

            # Random jitter
            jitter = np.random.normal(0, 0.02, point_cloud.shape).astype(np.float32)
            point_cloud = point_cloud + jitter

            # Random scale
            scale = np.random.uniform(0.8, 1.2)
            point_cloud = point_cloud * scale

        # Normalize again after augmentation
        point_cloud = point_cloud - point_cloud.mean(axis=0)
        point_cloud = point_cloud / (np.abs(point_cloud).max() + 1e-8)

        return torch.FloatTensor(point_cloud), label



def create_datasets(data_dir='./data', dataset='modelnet10', n_points=1024):
    """Download data and create train/test datasets and loaders."""
    modelnet_path = download_modelnet(data_dir=data_dir, dataset=dataset)

    train_dataset = ModelNetDataset(
        root_dir=modelnet_path, split='train', n_points=n_points,
        data_augmentation=True, dataset=dataset
    )
    test_dataset = ModelNetDataset(
        root_dir=modelnet_path, split='test', n_points=n_points,
        data_augmentation=False, dataset=dataset
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    return train_dataset, test_dataset, train_loader, test_loader
