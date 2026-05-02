"""Run complete R-SHAP analysis on ModelNet10 with PointNet."""

import os
import torch
import numpy as np

from src.data import create_datasets
from src.model import PointNet
from src.rshap import FixedRegionSHAP
from src.protocols import (
    diagnostic_value_function_signal,
    protocol_1_synthetic_manifolds,
    protocol_1_real_data,
    protocol_1_per_class,
    protocol_2,
    protocol_3_fixed,
    protocol_4,
    protocol_5,
    protocol_7,
    protocol_8,
    protocol_10_reference_comparison,
    protocol_11_critical_points,
    protocol_12,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_dataset, test_dataset, train_loader, test_loader = create_datasets()

    # Load trained model
    n_classes = len(train_dataset.classes)
    model = PointNet(n_classes=n_classes, use_tnet=True).to(device)

    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Output directory
    output_dir = 'results/'
    os.makedirs(output_dir, exist_ok=True)

    # Diagnostic: find best configuration
    best_config, _ = diagnostic_value_function_signal(model, test_dataset, device)
    print(f"Primary config: ref={best_config[0]}, space={best_config[1]}")

    # Protocol 1: GIR Manifold Preservation
    print("\n--- Protocol 1: GIR Manifold Preservation ---")
    protocol_1_synthetic_manifolds()
    protocol_1_real_data()
    protocol_1_per_class(test_dataset, device)

    # Protocol 2: Region Count Selection
    print("\n--- Protocol 2: Region Count Selection ---")
    protocol_2(model, test_dataset, device,
               ref=best_config[0], vs=best_config[1])

    # Protocol 3: Clustering Algorithm Impact
    print("\n--- Protocol 3: Clustering Impact ---")
    protocol_3_fixed(model, test_dataset, device,
                     ref=best_config[0], vs=best_config[1])

    # Protocol 4: Faithfulness
    print("\n--- Protocol 4: Faithfulness ---")
    protocol_4(model, test_dataset, device,
               ref=best_config[0], vs=best_config[1],
               output_dir=output_dir)

    # Protocol 5: Gradient Alignment
    print("\n--- Protocol 5: Gradient Alignment ---")
    protocol_5(model, test_dataset, device,
               ref=best_config[0], vs=best_config[1],
               output_dir=output_dir)

    # Protocol 7: Qualitative Evaluation
    print("\n--- Protocol 7: Qualitative ---")
    protocol_7(model, test_dataset, device,
               ref=best_config[0], vs=best_config[1],
               output_dir=output_dir)

    # Protocol 8: Per-Class Statistics
    print("\n--- Protocol 8: Statistics ---")
    protocol_8(model, test_dataset, device,
               ref=best_config[0], vs=best_config[1],
               output_dir=output_dir)

    # Protocol 10: Reference Comparison
    print("\n--- Protocol 10: Reference Comparison ---")
    protocol_10_reference_comparison(model, test_dataset, device,
                                     output_dir=output_dir)

    # Protocol 11: Critical Point Analysis
    print("\n--- Protocol 11: Critical Points ---")
    protocol_11_critical_points(model, test_dataset, device,
                                output_dir=output_dir)

    # Protocol 12: Occlusion Diagnostic
    print("\n--- Protocol 12: Occlusion Diagnostic ---")
    protocol_12(model, test_dataset, device,
                ref=best_config[0], vs=best_config[1],
                output_dir=output_dir)

    print("\n" + "=" * 60)
    print("R-SHAP analysis complete. Results saved to:", output_dir)


if __name__ == '__main__':
    main()
