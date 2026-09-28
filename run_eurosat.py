import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

from mer_core import (
    SmallCNN_Standard, SmallCNN_InNetwork, 
    ResNet18_Standard, ResNet18_InNetwork, 
    train_experiment
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Executando no dispositivo: {device}')

    in_channels = 3
    num_classes = 10

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    data_dir = './data'
    full_dataset = EuroSAT(root=data_dir, transform=transform, download=True)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    epochs_eval = [5, 10, 15, 20]

    model_factories = {
        'SmallCNN_Standard': SmallCNN_Standard,
        'SmallCNN_InNetwork': SmallCNN_InNetwork,
        'ResNet18_Standard': ResNet18_Standard,
        'ResNet18_InNetwork': ResNet18_InNetwork,
    }

    raw_results = {name: {seed: {} for seed in seeds} for name in model_factories}

    print('\n' + '=' * 80)
    print(f'INICIANDO EXPERIMENTO EUROSAT 128x128 ({len(seeds)} SEMENTES | {", ".join(map(str, epochs_eval))} ÉPOCAS)')
    print('=' * 80)

    for seed in seeds:
        print(f'\n>>>> SEMENTE {seed} <<<<')
        for name, model_cls in model_factories.items():
            t0 = time.time()
            res = train_experiment(
                model_cls=model_cls,
                train_loader=train_loader,
                test_loader=test_loader,
                target_epochs=epochs_eval,
                seed=seed,
                device=device,
                in_channels=in_channels,
                num_classes=num_classes,
            )
            raw_results[name][seed] = res
            t_elapsed = time.time() - t0

            metrics_str = ' | '.join([f"E{ep} F1: {res[ep]['f1']:.4f}" for ep in epochs_eval])
            print(f'[{name:<20}] Tempo: {t_elapsed:5.1f}s | {metrics_str}')

    output_filename = 'results_eurosat_128x128_8seeds.json'
    with open(output_filename, 'w') as f:
        json.dump(raw_results, f, indent=4)

    print('\n' + '=' * 80)
    print('RESUMO FINAL EUROSAT 128x128 - MÉDIA ± DESVIO PADRÃO (N=8 SEMENTES)')
    print('=' * 80)

    for ep in epochs_eval:
        print(f'\n--- ÉPOCAS: {ep} ---')
        print(f"{'Modelo':<22} | {'Macro-F1 (Média ± DP)':<25} | {'Acurácia (Média ± DP)':<25}")
        print('-' * 75)
        for name in model_factories:
            f1_list = [raw_results[name][s][ep]['f1'] for s in seeds]
            acc_list = [raw_results[name][s][ep]['acc'] for s in seeds]

            f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
            acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)

            print(f'{name:<22} | {f1_mean:.4f} ± {f1_std:.4f}          | {acc_mean:.4f} ± {acc_std:.4f}')

    print(f"\n[OK] Resultados brutos salvos com sucesso em '{output_filename}'.")