import torch


class MapeLoss(torch.nn.Module):
    def __init__(self):
        super(MapeLoss, self).__init__()
        return

    def forward(self, y_pred, y):
        return torch.mean((y - y_pred).abs() / torch.mean(torch.clip(y, 1e-6, 1).abs()))

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, y_pred, y):
        return torch.sqrt(self.mse(y_pred, y))

class RSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y):
        target_mean = torch.mean(y)
        ss_tot = torch.sum((y - target_mean) ** 2)
        ss_res = torch.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
