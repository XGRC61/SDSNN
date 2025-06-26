import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams

# Set font for plots
config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False,
}
rcParams.update(config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Mohr-Coulomb failure criterion function
def g(x, y, m, n):
    sin_phi = torch.sin(n / 180 * torch.pi)
    denom = torch.clamp(1 - sin_phi, min=1e-3)
    Nf = (1 + sin_phi) / denom
    MC = (x + y) / 2 * Nf + 2 * m * torch.sqrt(torch.clamp(Nf, min=1e-6))
    return MC

# PINN model definition
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, y, m, n):
        inputs = torch.cat((x, y, m, n), dim=1)
        return self.net(inputs)

# Custom loss function
def pinn_loss(model, x, y, m, n, true_z, training=True):
    z = model(x, y, m, n)

    if not torch.isfinite(z).all():
        return [torch.tensor(float('nan'), device=x.device)] * 7

    data_loss = torch.mean((z - true_z) ** 2)

    if not training:
        return [data_loss] + [torch.tensor(0.0, device=x.device)] * 6

    z_x = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    z_y = torch.autograd.grad(z.sum(), y, create_graph=True)[0]
    z_xx = torch.autograd.grad(z_x.sum(), x, create_graph=True)[0]
    physics_loss1 = torch.mean(torch.relu(-z_y) ** 2)
    physics_loss2 = torch.mean(torch.relu(z_xx))
    physics_loss3 = torch.tensor(0.0, device=x.device)

    # Constraint loss with tolerance
    x_denorm = x * x_std + x_mean
    z_denorm = z * z_std + z_mean
    strict_violation_mask = z_denorm < (1.0 * x_denorm)
    violation = x_denorm[strict_violation_mask] - z_denorm[strict_violation_mask]

    if violation.numel() > 0 and torch.isfinite(violation).all():
        violation_scaled = torch.clamp(violation / z_std, max=0.005)
        constraint_loss = torch.mean(torch.relu(violation_scaled))
    else:
        constraint_loss = torch.tensor(0.0, device=x.device)

    # Left boundary loss (when x == y)
    mask = torch.abs(x - y) < 1e-6
    if torch.any(mask):
        z_b = z[mask]
        g_b = g(x[mask], y[mask], m[mask], n[mask])
        boundary_loss = torch.mean((z_b - g_b) ** 2) if torch.isfinite(g_b).all() else torch.tensor(0.0, device=x.device)
    else:
        boundary_loss = torch.tensor(0.0, device=x.device)

    # Right boundary loss
    mask_rb = torch.abs(x - true_z) < 0.02
    if torch.any(mask_rb):
        x_rb = x[mask_rb]
        y_rb = y[mask_rb]
        m_rb = m[mask_rb]
        n_rb = n[mask_rb]
        if all(t.ndim == 2 for t in [x_rb, y_rb, m_rb, n_rb]) and x_rb.shape[0] > 0:
            z_rb = model(x_rb, y_rb, m_rb, n_rb)
            right_boundary_loss = torch.mean((z_rb - x_rb) ** 2)
        else:
            right_boundary_loss = torch.tensor(0.0, device=x.device)
    else:
        right_boundary_loss = torch.tensor(0.0, device=x.device)

    return data_loss, physics_loss1, physics_loss2, physics_loss3, boundary_loss, constraint_loss, right_boundary_loss

# Loss ratio computation
def compute_loss_ratio(loss_now, loss_prev, max_clip=5, min_val=1e-6):
    ratio = loss_now / np.clip(loss_prev, min_val, None)
    ratio = np.clip(ratio, 0, np.exp(max_clip))
    return ratio

# Load data
data = pd.read_csv('data/datatrain.csv')
datatest = pd.read_csv('data/datatest.csv')
print(f"Train size: {len(data)} samples | Test size: {len(datatest)} samples")

# Normalization
def normalize(v):
    return (v - v.mean()) / v.std(), v.mean(), v.std()

def to_tensor(v):
    return torch.tensor(v.values if hasattr(v, 'values') else v, dtype=torch.float32).view(-1, 1)

# Prepare training and validation data
x_train, x_mean, x_std = normalize(to_tensor(data['sig2']))
y_train, y_mean, y_std = normalize(to_tensor(data['sig3']))
m_train, m_mean, m_std = normalize(to_tensor(data['cohesion']))
n_train, n_mean, n_std = normalize(to_tensor(data['friction']))
z_train = to_tensor(data['sig1'])
z_mean, z_std = z_train.mean(), z_train.std()
z_train_norm = (z_train - z_mean) / z_std

x_valid = ((to_tensor(datatest['sig2']) - x_mean) / x_std).to(device)
y_valid = ((to_tensor(datatest['sig3']) - y_mean) / y_std).to(device)
m_valid = ((to_tensor(datatest['cohesion']) - m_mean) / m_std).to(device)
n_valid = ((to_tensor(datatest['friction']) - n_mean) / n_std).to(device)
z_valid = ((to_tensor(datatest['sig1']) - z_mean) / z_std).to(device)

x_train = x_train.requires_grad_().to(device)
y_train = y_train.requires_grad_().to(device)
m_train = m_train.requires_grad_().to(device)
n_train = n_train.requires_grad_().to(device)
z_train_norm = z_train_norm.to(device)

# Initialize model and optimizer
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training loop
losses, valloss = [], []
num_epochs = 40000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    losses_now = pinn_loss(model, x_train, y_train, m_train, n_train, z_train_norm, training=True)

    if not all(torch.isfinite(loss) for loss in losses_now):
        print(f"NaN at epoch {epoch}")
        continue

    if epoch == 0:
        prev_losses = [l.item() for l in losses_now]
        weights = torch.ones(len(losses_now), dtype=torch.float32).to(device)
    elif epoch % 100 == 0:
        prev_losses = losses[-1]
        etheta = [compute_loss_ratio(l.item(), p) for l, p in zip(losses_now, prev_losses)]
        weights = torch.tensor(etheta, dtype=torch.float32).to(device)
        weights = weights / torch.max(weights)

    total_loss = sum(w * l for w, l in zip(weights, losses_now))
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    losses.append([l.item() for l in losses_now])

    model.eval()
    val_losses_now = pinn_loss(model, x_valid, y_valid, m_valid, n_valid, z_valid, training=False)
    val_total_loss = val_losses_now[0]
    valloss.append(val_total_loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Train Loss: {total_loss.item():.4f} | Val Loss: {val_total_loss.item():.4f}")

# Save model
torch.save(model.state_dict(), 'trained_SDSNN.pth')
print("✅ Model saved as trained_SDSNN.pth")

