import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import copy
from tqdm import tqdm

def generate_torus(major_radius, minor_radius, center, rotation_angles, n_samples=500):
    # Sample theta and phi angles from uniform distributions
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)

    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta) + center[0]
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta) + center[1]
    z = minor_radius * np.sin(phi) + center[2]

    # Apply rotation
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]
    ])

    rotation_y = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
        [0, 1, 0],
        [ -np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]
    ])

    rotation_z = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
        [0, 0, 1]
    ])

    points = np.vstack((x, y, z))
    rotated_points = rotation_x @ rotation_y @ rotation_z @ points
    
    return rotated_points

def generate_interwined_tori(n_samples=500, visualize=False):

    # Define parameters for the tori
    major_radius = 5
    minor_radius = 1.5

    # Generate the first torus at the origin
    torus1_points = generate_torus(major_radius, minor_radius, center=[0, 0, 0], rotation_angles=[0, 0, 0], n_samples=n_samples)

    # Generate the second torus, rotated and translated to pass through the hole of the first torus
    torus2_points = generate_torus(major_radius, minor_radius, center=[0, 5, 0], rotation_angles=[0, np.pi / 2, 0], n_samples=n_samples)

    if visualize:
        # Combine the points
        all_points = np.hstack((torus1_points, torus2_points))

        # Convert the points to a DataFrame for easier plotting with Plotly
        df = pd.DataFrame({
            'x': all_points[0],
            'y': all_points[1],
            'z': all_points[2],
        })

        # Create a 3D scatter plot using Plotly Express
        fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.3)
        fig.update_traces(marker=dict(size=2))  # Set the marker size to 1 or any desired small value

        # Show the plot
        fig.show()

    return torus1_points, torus2_points

def generate_tori_dataloader(n_samples=500, visualize=False):

    # Generate the tori
    torus1_points, torus2_points = generate_interwined_tori(n_samples=n_samples, visualize=visualize)

    # Labels
    labels1 = np.zeros(torus1_points.shape[1])
    labels2 = np.ones(torus2_points.shape[1])

    # Combine data and labels
    all_points = np.hstack((torus1_points, torus2_points)).T
    all_labels = np.hstack((labels1, labels2))

    # Convert to PyTorch tensors
    X = torch.tensor(all_points, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


def pca_projection_3d(X, y):
    # X is a 2D array of shape (n_samples, n_features)
    # y is a 1D array of shape (n_samples,)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(X)
    df_pca = pd.DataFrame(pca_features, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Label'] = y.numpy()
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Label', opacity=0.5)
    fig.update_traces(marker=dict(size=2))
    return fig

def train_model(model, criterion, optimizer, dataloader, epochs, print_every=None):
    model.train()
    device = next(model.parameters()).device
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            accuracy = 100 * correct / total

        total_loss /= total
        if print_every is not None:
            if epoch == 0 or (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Training Accuracy: {accuracy:.2f}%")

    return model, total_loss, accuracy


# Define a function to compute the accuracy
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X, batch_y
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return 100 * correct / total

# TODO: batch weights are not copied
def generate_random_direction_old(model):
    random_direction = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'bn' not in name and 'downsample' not in name:
            random_filter = torch.randn_like(param.data)
            
            # bias
            if len(param.shape) == 1:
                normalized_filter =  param.data
            # Check if the parameter is a weight matrix of a dense layer (2D)
            if len(param.shape) == 2:
                # Normalize each row (filter) using the Frobenius norm
                normalized_filter = random_filter / torch.norm(random_filter, dim=1, keepdim=True)
                # Multiply by the norm of the original filter
                normalized_filter *= torch.norm(param.data, dim=1, keepdim=True)

            if len(param.shape) == 4:
                # For convolutional layers (e.g., 4D shape), normalize each filter separately
                normalized_filter = torch.zeros_like(random_filter)
                for i in range(random_filter.shape[0]):
                    for j in range(random_filter.shape[1]):
                        filter_norm = torch.norm(random_filter[i][j])
                        normalized_filter[i][j] = random_filter[i][j] / filter_norm
                        normalized_filter[i][j] *= torch.norm(param.data[i][j])

            # Store the normalized random direction
            random_direction[name] = normalized_filter
        else:
            random_direction[name] = param.data


    random_direction_scaled = {key: weight * 1. for key, weight in random_direction.items()}  
    return random_direction_scaled

def generate_random_direction(model):
    
    batch_norm_keys = set()
    # Traverse the model's modules to find Batch Normalization layers
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, _ in module.named_parameters(recurse=False):
                full_key_name = f"{name}.{param_name}"
                batch_norm_keys.add(full_key_name)
            for buf_name, _ in module.named_buffers(recurse=False):
                full_key_name = f"{name}.{buf_name}"
                batch_norm_keys.add(full_key_name) 

    random_direction = {}
    state_dict = model.state_dict()

    for key, param in tqdm(state_dict.items()):
        if key in batch_norm_keys or 'downsample' in key:
            random_direction[key] = param.data
        else:
            # bias
            if len(param.shape) == 1:
                normalized_filter =  param.data
            # Check if the parameter is a weight matrix of a dense layer (2D)
            if len(param.shape) == 2:
                random_filter = torch.randn_like(param.data)
                # Normalize each row (filter) using the Frobenius norm
                normalized_filter = random_filter / torch.norm(random_filter, dim=1, keepdim=True)
                # Multiply by the norm of the original filter
                normalized_filter *= torch.norm(param.data, dim=1, keepdim=True)
            if len(param.shape) == 4:
                # For convolutional layers (e.g., 4D shape), normalize each filter separately
                random_filter = torch.randn_like(param.data)
                normalized_filter = torch.zeros_like(random_filter)
                for i in range(random_filter.shape[0]):
                    for j in range(random_filter.shape[1]):
                        filter_norm = torch.norm(random_filter[i][j]) + 1e-10
                        normalized_filter[i][j] = random_filter[i][j] / filter_norm
                        normalized_filter[i][j] *= torch.norm(param.data[i][j])
            # Store the normalized random direction
            random_direction[key] = normalized_filter
    
    return random_direction

def generate_random_directions(model):
    '''
    Currently, this function generates only two random directions.
    '''
    print('generating random directions...')
    batch_norm_keys = set()
    # Traverse the model's modules to find Batch Normalization layers
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, _ in module.named_parameters(recurse=False):
                full_key_name = f"{name}.{param_name}"
                batch_norm_keys.add(full_key_name)
            for buf_name, _ in module.named_buffers(recurse=False):
                full_key_name = f"{name}.{buf_name}"
                batch_norm_keys.add(full_key_name) 

    random_direction1 = {}
    random_direction2 = {}
    state_dict = model.state_dict()

    for key, param in tqdm(state_dict.items()):
        if key in batch_norm_keys or 'downsample' in key:
            random_direction1[key] = param.data
            random_direction2[key] = param.data
        else:
            # bias
            if len(param.shape) == 1:
                normalized_filter1 =  param.data
                normalized_filter2 =  param.data
            # Check if the parameter is a weight matrix of a dense layer (2D)
            elif len(param.shape) == 2:
                random_filter1 = torch.randn_like(param.data)
                random_filter2 = torch.randn_like(param.data)
                # Normalize each row (filter) using the Frobenius norm
                normalized_filter1 = random_filter1 / torch.norm(random_filter1, dim=1, keepdim=True)
                normalized_filter2 = random_filter2 / torch.norm(random_filter2, dim=1, keepdim=True)
                # Multiply by the norm of the original filter
                original_filter_norm = torch.norm(param.data, dim=1, keepdim=True)
                normalized_filter1 *= original_filter_norm
                normalized_filter2 *= original_filter_norm
            elif len(param.shape) == 4:
                # For convolutional layers (e.g., 4D shape), normalize each filter separately
                random_filter1 = torch.randn_like(param.data)
                random_filter2 = torch.randn_like(param.data)
                normalized_filter1 = torch.zeros_like(random_filter1)
                normalized_filter2 = torch.zeros_like(random_filter2)
                for i in range(random_filter1.shape[0]):
                    for j in range(random_filter1.shape[1]):
                        filter_norm1 = torch.norm(random_filter1[i][j])
                        filter_norm2 = torch.norm(random_filter2[i][j])
                        normalized_filter1[i][j] = random_filter1[i][j] / filter_norm1
                        normalized_filter2[i][j] = random_filter2[i][j] / filter_norm2
                        original_filter_norm = torch.norm(param.data[i][j])
                        normalized_filter1[i][j] *= torch.norm(param.data[i][j])
                        normalized_filter2[i][j] *= torch.norm(param.data[i][j])
            
            # Store the normalized random direction
            random_direction1[key] = normalized_filter1
            random_direction2[key] = normalized_filter2
    print('done')
    return random_direction1, random_direction2

# Define a function to calculate the loss landscape
def calculate_loss_landscape(model, criterion, dataloader, direction1, direction2, 
                             grid_size=21, grid_range=(-1, 1), percentage=1.):
    model.eval()

    batch_norm_keys = set()
    # Traverse the model's modules to find Batch Normalization layers
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, _ in module.named_parameters(recurse=False):
                full_key_name = f"{name}.{param_name}"
                batch_norm_keys.add(full_key_name)
            for buf_name, _ in module.named_buffers(recurse=False):
                full_key_name = f"{name}.{buf_name}"
                batch_norm_keys.add(full_key_name) 

    with torch.no_grad():
        num_batches_for_test = int(len(dataloader)*percentage)
        losses = []
        device = next(model.parameters()).device
        # Save the original weights
        original_weights = copy.deepcopy(model.state_dict())

        for x in np.linspace(grid_range[0], grid_range[1], grid_size):
            for y in np.linspace(grid_range[0], grid_range[1], grid_size):
                w = copy.deepcopy(original_weights)
                for key in original_weights.keys():
                    # Update only if the key is not in batch_norm_keys and 'downsample' is not in key
                    if key not in batch_norm_keys and 'downsample' not in key:
                        direction1_value = direction1[key].to(dtype=w[key].dtype, device=device)
                        direction2_value = direction2[key].to(dtype=w[key].dtype, device=device)
                        w[key] += x * direction1_value + y * direction2_value

                model.load_state_dict(w)
                loss = 0.
                total = 0
                for idx, (images, labels) in enumerate(dataloader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    batch_loss = criterion(outputs, labels)
                    if torch.any(torch.isnan(batch_loss)):
                        print("batch_loss contains NaN values in [calculate_loss_landscape]")
                    loss += batch_loss.item()
                    total += len(labels)
                    if idx == num_batches_for_test:
                        break

                losses.append((x, y, loss/total))

        model.load_state_dict(original_weights)
    return losses


def visualize_loss_landscape(model, criterion, dataloader, direction1, direction2, grid_size=11, grid_range=(-1, 1), maximum_loss = 10, percentage=0.4):

    # Calculate the loss landscape
    losses = calculate_loss_landscape(model, criterion, dataloader, direction1, direction2, grid_size, grid_range, percentage)

    # Plot the loss landscape
    losses = np.array(losses)

    x = losses[:, 0]
    y = losses[:, 1]
    z = losses[:, 2]

    if maximum_loss is None:
        maximum_loss = np.max(z)
    else:   
        for i in range(len(z)):
            if z[i] > maximum_loss:
                z[i] = maximum_loss

    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = z.reshape(len(np.unique(x)), len(np.unique(y)))

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Loss Landscape', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90),
                    scene=dict(zaxis=dict(range=[0, maximum_loss])))
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=.5),
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[0, maximum_loss])
        )
    )
    # center = int((grid_size**2 - 1)/2)
    # print(f'test loss : {z[center]}')
    # print(f'max loss : {np.max(z)}')
    fig.show()
    return fig

def visualize_features(model, dataloader, loss, accuracy, interactive_mode=False):
    model.eval()
    X = dataloader.dataset.tensors[0]
    y = dataloader.dataset.tensors[1]
    n_layers = int(len(model.layers)/2)
    feature_sets = []
    with torch.no_grad():
        for i in range(n_layers):
            features = model.extract_features(X, layer_num=i).detach().numpy()
            feature_sets.append(features)

    # Create subplots with 1 row and 6 columns (always 3D scatter plots)
    fig = make_subplots(rows=1, cols=4, specs=[[{'type': 'scatter3d'}] * 4])

    # Create scatter plots for each feature set
    for i, features in enumerate(feature_sets):
        scatter_fig = pca_projection_3d(features, y) # 3D projection function
        for trace in scatter_fig['data']:
            fig.add_trace(trace, row=1, col=i+1)

    fig.add_annotation(
        text=f"loss : {loss:.4f} (acc : {accuracy})", # Replace with the desired text
        xref="paper", yref="paper",
        x=0, y=1,
        showarrow=False,
        font=dict(size=14)
    )

    fig.update_layout(height=350, width=1800)
    
    # Show figure with or without interactive mode
    if interactive_mode:
        fig.show()
    else:
        fig.show(renderer="png") # Change to static rendering