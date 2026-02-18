import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, act_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        with torch.no_grad():
            logits = self.forward(torch.tensor(obs, dtype=torch.float32))
            return torch.argmax(logits).item()


# --- Behavior Cloning Trainer ---
def train_policy(obs, acts):
    print(f"Training on {len(obs)} samples with neural network...")
    if len(obs) == 0:
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        return DummyModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    acts = torch.tensor(acts, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(obs, acts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = PolicyNet(obs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return model
