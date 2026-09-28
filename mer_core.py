import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import f1_score, accuracy_score

# ==========================================
# 1. Configuração de Reprodutibilidade
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. Bloco Morfológico (Estável)
# ==========================================
class LocalExtinctionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, base_kernel=3, ranks=2):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.base_kernel = base_kernel
        self.ranks = ranks
        total_ext_channels = in_channels * 6 * ranks

        self.proj = nn.Conv2d(
            total_ext_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        ext_maps = []
        eps = 1e-8

        for r in range(1, self.ranks + 1):
            k = self.base_kernel * r
            if k % 2 == 0:
                k += 1
            padding = k // 2

            dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=padding)
            ero = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=padding)
            ope = F.max_pool2d(ero, kernel_size=k, stride=1, padding=padding)
            clo = -F.max_pool2d(-dil, kernel_size=k, stride=1, padding=padding)

            eh_pos = x - ero
            eh_neg = dil - x
            ea_pos = x - ope
            ea_neg = clo - x

            ev_pos = torch.sqrt(torch.clamp(eh_pos * ea_pos, min=0.0) + eps)
            ev_neg = torch.sqrt(torch.clamp(eh_neg * ea_neg, min=0.0) + eps)

            ext_maps.extend([eh_pos, eh_neg, ea_pos, ea_neg, ev_pos, ev_neg])

        out = torch.cat(ext_maps, dim=1)
        out = self.proj(out)
        out = self.bn(out)
        return self.act(out + x)

# ==========================================
# 3. Modelos: SmallCNN e ResNet-18
# ==========================================
class SmallCNN_Standard(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class SmallCNN_InNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.ext_block = LocalExtinctionBlock2D(
            in_channels=32, out_channels=32, base_kernel=3, ranks=2
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ext_block(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNet18_Standard(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18_InNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)

        conv1 = (
            base.conv1
            if in_channels == 3
            else nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        )

        self.stem = nn.Sequential(conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.morph_block = LocalExtinctionBlock2D(
            in_channels=64, out_channels=64, base_kernel=3, ranks=2
        )
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.morph_block(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 4. Avaliação e Treinamento
# ==========================================
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.view(-1).numpy())

    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds)
    return macro_f1, acc

def train_experiment(
    model_cls, train_loader, test_loader, target_epochs, seed, device, in_channels=3, num_classes=10
):
    set_seed(seed)
    model = model_cls(in_channels=in_channels, num_classes=num_classes).to(device)

    lr = 3e-4 if 'ResNet' in model_cls.__name__ else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    max_epochs = max(target_epochs)
    eval_checkpoints = target_epochs
    results = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.view(-1).to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch in eval_checkpoints:
            f1, acc = evaluate(model, test_loader, device)
            results[epoch] = {'f1': f1, 'acc': acc}

    # Limpeza de memória ao final de cada semente/modelo
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return results