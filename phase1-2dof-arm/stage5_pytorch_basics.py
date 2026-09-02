import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


net = SimpleNet().to(device)
print(net)

torch.manual_seed(0)
x_train = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
y_train = 2 * x_train + 1 + 0.1 * torch.randn_like(x_train)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = net(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch={epoch} loss={loss.item():.4f}")

test_x = torch.tensor([[3.0]]).to(device)
print(f"pred for x=3 (want ~7): {net(test_x).item():.3f}")

torch.save(net.state_dict(), "simple_net.pth")

net2 = SimpleNet().to(device)
net2.load_state_dict(torch.load("simple_net.pth", weights_only=True))
net2.eval()
print(f"pred from loaded model: {net2(test_x).item():.3f}")
