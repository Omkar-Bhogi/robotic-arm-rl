import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=6, action_dim=2, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork().to(device)
    print(policy)

    dummy_obs = torch.randn(5, 6).to(device)
    actions = policy(dummy_obs)

    print(f"in: {dummy_obs.shape}  out: {actions.shape}")
    print(actions)
    print("all in [-1,1]:", torch.all((actions >= -1) & (actions <= 1)).item())
