"""Experimental protocols for R-SHAP validation."""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .rshap import GeometricInterpolationReference, RegionSHAP, RegionSegmentation


def diagnostic_value_function_signal(model, test_dataset, device, n_regions=8):
    """
    Diagnostic: Check value function signal across reference mechanisms.
    GIR is the primary reference.
    Zero/Mean are tested as ablation comparisons.
    """

    ref_mechanisms = ['gir', 'zero', 'mean']
    value_spaces = ['logit', 'prob']

    class_samples = []
    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                class_samples.append((pc.numpy(), class_idx, test_dataset.classes[class_idx]))
                break

    print(f"\nTesting {len(class_samples)} classes x {len(ref_mechanisms)} references x {len(value_spaces)} spaces")
    print(f"Regions: M = {n_regions}\n")

    all_signals = {}

    for ref in ref_mechanisms:
        for vs in value_spaces:
            explainer = RegionSHAP(
                model=model, reference_mechanism=ref,
                n_regions=n_regions, n_samples=100, device=device,
                value_space=vs
            )

            all_drops = []
            for pc, class_idx, class_name in class_samples:
                regions = explainer.segment_regions(pc)
                drops = explainer.single_region_occlusion(pc, regions, class_idx)
                std = np.std(drops)
                max_drop = np.max(np.abs(drops))
                all_drops.append((class_name, std, max_drop, drops))

            avg_std = np.mean([d[1] for d in all_drops])
            avg_max = np.mean([d[2] for d in all_drops])
            signal_strength = avg_std * avg_max

            marker = " [PRIMARY]" if ref == 'gir' and vs == 'logit' else ""
            print(f"  ref={ref:5s} space={vs:5s} | Avg Std={avg_std:.4f}, Avg MaxDrop={avg_max:.4f}, Signal={signal_strength:.6f}{marker}")
            all_signals[(ref, vs)] = (signal_strength, all_drops)

    # GIR + logit is the primary configuration per theoretical framework
    primary_config = ('gir', 'logit')
    primary_signal, primary_drops = all_signals[primary_config]

    # Also report which config had strongest raw signal for comparison
    best_key = max(all_signals, key=lambda k: all_signals[k][0])
    if best_key != primary_config:
        print(f"\n  ref={best_key[0]}, space={best_key[1]} had strongest raw signal ({all_signals[best_key][0]:.6f})")

    print(f"\n  Per-class drops (GIR + logit):")
    for class_name, std, max_drop, drops in primary_drops:
        drops_str = ", ".join([f"{d:+.4f}" for d in drops])
        print(f"    {class_name:12s}: std={std:.4f}, max={max_drop:.4f} | [{drops_str}]")

    return primary_config, primary_drops

best_config, best_drops = diagnostic_value_function_signal(model, test_dataset, device)


def protocol_1_synthetic_manifolds():
    """
    Protocol 1: Synthetic manifold (sphere).
    """

    n_points = 1024
    results = {}

    # Test on sphere
    print("\n1. Testing on unit sphere...")
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere = np.stack([x, y, z], axis=1)

    # Remove 50% of points randomly
    present_mask = np.random.rand(n_points) > 0.5

    # Test different methods
    methods = {
        'GIR': GeometricInterpolationReference(k_neighbors=20),
        'Zero': None,
        'Random': None
    }

    sphere_results = {}

    for method_name, gir in methods.items():
        if method_name == 'GIR':
            perturbed = gir.perturb(sphere, present_mask)
        elif method_name == 'Zero':
            perturbed = sphere.copy()
            perturbed[~present_mask] = 0.0
        else:  # Random
            perturbed = sphere.copy()
            perturbed[~present_mask] = np.random.randn(np.sum(~present_mask), 3) * 0.1

        # Compute distance to manifold (unit sphere)
        interpolated_points = perturbed[~present_mask]
        distances_to_manifold = np.abs(np.linalg.norm(interpolated_points, axis=1) - 1.0)

        mean_dist = distances_to_manifold.mean()
        max_dist = distances_to_manifold.max()

        sphere_results[method_name] = {
            'mean_distance': mean_dist,
            'max_distance': max_dist
        }

        print(f"  {method_name:10s}: Mean dist = {mean_dist:.4f}, Max dist = {max_dist:.4f}")

    results['sphere'] = sphere_results

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={'projection': '3d'})

    for idx, (method_name, gir) in enumerate(methods.items()):
        ax = axes[idx]

        if method_name == 'GIR':
            perturbed = gir.perturb(sphere, present_mask)
        elif method_name == 'Zero':
            perturbed = sphere.copy()
            perturbed[~present_mask] = 0.0
        else:
            perturbed = sphere.copy()
            perturbed[~present_mask] = np.random.randn(np.sum(~present_mask), 3) * 0.1

        # Plot present points
        ax.scatter(perturbed[present_mask, 0], perturbed[present_mask, 1],
                  perturbed[present_mask, 2], c='blue', s=5, alpha=0.6, label='Present')

        # Plot interpolated points
        ax.scatter(perturbed[~present_mask, 0], perturbed[~present_mask, 1],
                  perturbed[~present_mask, 2], c='red', s=10, alpha=0.8, label='Interpolated')

        ax.set_title(f'{method_name}\nMean Dist: {sphere_results[method_name]["mean_distance"]:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_facecolor('white')
        ax.grid(True)
        ax.legend()


    plt.tight_layout()
    plt.savefig(f'{output_directory}protocol1_sphere_manifold.png', dpi=600, bbox_inches='tight')
    plt.show()

    return results

# Run Protocol 1
protocol1_results = protocol_1_synthetic_manifolds()


def protocol_1_real_data():
    """
    Protocol 1: GIR on real point cloud data.
    """

    # Get a sample from test set
    sample_pc, label = test_dataset[0]
    sample_pc = sample_pc.numpy()

    print(f"\nTesting on real point cloud (class: {train_dataset.classes[label]})")

    # Remove 50% of points
    n_points = sample_pc.shape[0]
    present_mask = np.random.rand(n_points) > 0.5

    gir = GeometricInterpolationReference(k_neighbors=20)

    # Test GIR
    perturbed_gir = gir.perturb(sample_pc, present_mask)

    # Compute Chamfer distance
    dist_matrix = cdist(perturbed_gir, sample_pc)
    chamfer_dist = dist_matrix.min(axis=1).mean()

    print(f"  Chamfer distance: {chamfer_dist:.4f}")
    print(f"  Points interpolated: {np.sum(~present_mask)}/{n_points}")

    # Visualize
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(sample_pc[:, 0], sample_pc[:, 1], sample_pc[:, 2], c='blue', s=10)
    ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.set_title('Original Point Cloud')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(sample_pc[present_mask, 0], sample_pc[present_mask, 1],
               sample_pc[present_mask, 2], c='blue', s=10, label='Present')
    ax2.set_facecolor('white')
    ax2.grid(False)
    ax2.set_title('50% Points Removed')
    ax2.legend()

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(perturbed_gir[present_mask, 0], perturbed_gir[present_mask, 1],
               perturbed_gir[present_mask, 2], c='blue', s=10, alpha=0.6, label='Present')
    ax3.scatter(perturbed_gir[~present_mask, 0], perturbed_gir[~present_mask, 1],
               perturbed_gir[~present_mask, 2], c='red', s=15, alpha=0.8, label='Interpolated (GIR)')
    ax3.set_facecolor('white')
    ax3.grid(False)
    ax3.set_title(f'GIR Interpolation\nChamfer: {chamfer_dist:.3f}')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'{output_directory}protocol1_real_data.png', dpi=600, bbox_inches='tight')
    plt.show()

    return chamfer_dist

# Run Protocol 1 (Real Data)
protocol1_real_results = protocol_1_real_data()


def protocol_1_per_class(test_dataset, device):
    """Protocol 1: GIR validation on one sample per class."""

    # Get one sample per class
    class_samples = []
    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            pc, label = test_dataset[i]
            if label == class_idx:
                class_samples.append((pc.numpy(), label, test_dataset.classes[label]))
                break

    results = {'class': [], 'chamfer': []}
    gir = GeometricInterpolationReference(k_neighbors=20)

    for pc, label, class_name in tqdm(class_samples, desc="Testing GIR"):
        # Remove 50% of points
        n_points = pc.shape[0]
        present_mask = np.random.rand(n_points) > 0.5

        # Interpolate
        perturbed = gir.perturb(pc, present_mask)

        # Chamfer distance
        dist_matrix = cdist(perturbed, pc)
        chamfer_dist = dist_matrix.min(axis=1).mean()

        results['class'].append(class_name)
        results['chamfer'].append(chamfer_dist)

        print(f"  {class_name:12s}: Chamfer = {chamfer_dist:.4f}")

    print(f"\n  Mean Chamfer: {np.mean(results['chamfer']):.4f}")

    return results

protocol1_per_class_results = protocol_1_per_class(test_dataset, device)


def protocol_2(model, test_dataset, device, ref=best_config[0], vs=best_config[1]):

    M_values = [4, 8, 12, 16, 20, 24]

    class_samples = []
    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                class_samples.append((pc.numpy(), class_idx, test_dataset.classes[class_idx]))
                break

    results = {M: [] for M in M_values}

    for M in M_values:
        t0 = time.time()
        for pc, class_idx, class_name in class_samples:
            explainer = RegionSHAP(
                model=model, reference_mechanism=ref,
                n_regions=M, n_samples=500, device=device, value_space=vs
            )
            importances, baseline, prediction, regions, _ = explainer.explain(pc, class_idx, verbose=False)
            results[M].append({
                'importances': importances,
                'std': np.std(importances),
                'range': np.max(importances) - np.min(importances),
                'class': class_name
            })
        elapsed = time.time() - t0
        avg_std = np.mean([r['std'] for r in results[M]])
        avg_range = np.mean([r['range'] for r in results[M]])
        print(f"  M={M:2d}: Time={elapsed:.2f}s, Avg Std={avg_std:.4f}, Avg Range={avg_range:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    stds = [np.mean([r['std'] for r in results[M]]) for M in M_values]
    ranges = [np.mean([r['range'] for r in results[M]]) for M in M_values]

    axes[0].bar(range(len(M_values)), stds, tick_label=M_values)
    axes[0].set_xlabel("Number of Regions (M)")
    axes[0].set_ylabel("Avg Importance Std")
    axes[0].set_title("Importance Variation by M")

    axes[1].bar(range(len(M_values)), ranges, tick_label=M_values, color='orange')
    axes[1].set_xlabel("Number of Regions (M)")
    axes[1].set_ylabel("Avg Importance Range")
    axes[1].set_title("Max-Min Range by M")

    # Show importance distributions for M=8
    for r in results[8]:
        axes[2].bar(range(8), r['importances'], alpha=0.3, label=r['class'])
    axes[2].set_xlabel("Region")
    axes[2].set_ylabel("Importance")
    axes[2].set_title("Importance Distribution (M=8)")

    plt.tight_layout()
    plt.savefig(f'{output_directory}protocol2.png', dpi=600, bbox_inches='tight')
    plt.show()

    return results

p2_results = protocol_2(model, test_dataset, device)


def protocol_3(model, test_dataset, device, n_regions=8, ref=best_config[0], vs=best_config[1]):

    sample_pc, label = test_dataset[0]
    sample_pc = sample_pc.numpy()
    label = label if isinstance(label, int) else label.item()

    methods = {}

    # FPS + Voronoi
    regions_fps, _ = RegionSegmentation.fps_voronoi(sample_pc, n_regions)
    methods['fps'] = regions_fps

    # K-Means
    regions_km, _ = RegionSegmentation.kmeans_clustering(sample_pc, n_regions)
    methods['kmeans'] = regions_km

    # Spectral
    regions_sp, _ = RegionSegmentation.spectral_clustering(sample_pc, n_regions)
    methods['spectral'] = regions_sp

    all_importances = {}
    for name, regions in methods.items():
        explainer = RegionSHAP(
            model=model, reference_mechanism=ref,
            n_regions=n_regions, n_samples=500, device=device, value_space=vs
        )
        # Override region segmentation with pre-computed regions
        baseline_pc = explainer.perturb_coalition(sample_pc, np.zeros(n_regions), regions)
        baseline = explainer.evaluate_model(baseline_pc, label)
        prediction = explainer.evaluate_model(sample_pc, label)

        coalitions = explainer.paired_coalition_sampling()
        values = []
        for coal in coalitions:
            perturbed = explainer.perturb_coalition(sample_pc, coal, regions)
            values.append(explainer.evaluate_model(perturbed, label))
        coalitions = np.array(coalitions)
        values = np.array(values)

        importances = explainer.solve_shapley_ols(coalitions, values, prediction, baseline)
        all_importances[name] = importances

        print(f"  {name:10s}: Max={np.max(importances):.4f}, Min={np.min(importances):.4f}, "
              f"Std={np.std(importances):.4f}, Range={np.max(importances)-np.min(importances):.4f}")

    # Clustering sensitivity
    all_phi = np.array(list(all_importances.values()))
    mean_phi = all_phi.mean(axis=0)
    sensitivity = np.mean([np.linalg.norm(phi - mean_phi) for phi in all_phi])
    print(f"\n  Clustering Sensitivity: {sensitivity:.4f}")

    return all_importances

p3_results = protocol_3(model, test_dataset, device)


def protocol_4(model, test_dataset, device, n_regions=8,
                     ref=best_config[0], vs=best_config[1],
                     output_dir=output_directory):
    """
    Protocol 4: Faithfulness with deletion/insertion curves.
    """

    os.makedirs(output_dir, exist_ok=True)

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=1000, device=device, value_space=vs
    )

    results = []
    skipped = []

    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                pc = pc.numpy()
                class_name = test_dataset.classes[class_idx]
                break

        importances, baseline_val, pred_val, regions, tc = explainer.explain(
            pc, class_idx, verbose=False
        )

        pred_prob = explainer.evaluate_model_prob(pc, class_idx)
        base_pc = explainer.perturb_coalition(pc, np.zeros(n_regions), regions)
        base_prob = explainer.evaluate_model_prob(base_pc, class_idx)

        delta_f = abs(pred_prob - base_prob)

        # Deletion curve
        del_order = np.argsort(-importances)
        del_curve = [pred_prob]
        for k in range(1, n_regions + 1):
            coalition = np.ones(n_regions)
            coalition[del_order[:k]] = 0
            perturbed = explainer.perturb_coalition(pc, coalition, regions)
            del_curve.append(explainer.evaluate_model_prob(perturbed, class_idx))

        # Insertion curve
        ins_curve = [base_prob]
        for k in range(1, n_regions + 1):
            coalition = np.zeros(n_regions)
            coalition[del_order[:k]] = 1
            perturbed = explainer.perturb_coalition(pc, coalition, regions)
            ins_curve.append(explainer.evaluate_model_prob(perturbed, class_idx))

        audc = np.mean(del_curve) / delta_f
        auic = np.mean(ins_curve) / delta_f
        faithfulness = auic - audc

        # Random baseline
        rng = np.random.RandomState(0)
        rand_order = rng.permutation(n_regions)
        rand_del = [pred_prob]
        rand_ins = [base_prob]
        for k in range(1, n_regions + 1):
            coal_d = np.ones(n_regions); coal_d[rand_order[:k]] = 0
            rand_del.append(explainer.evaluate_model_prob(
                explainer.perturb_coalition(pc, coal_d, regions), class_idx))
            coal_i = np.zeros(n_regions); coal_i[rand_order[:k]] = 1
            rand_ins.append(explainer.evaluate_model_prob(
                explainer.perturb_coalition(pc, coal_i, regions), class_idx))
        rand_f = np.mean(rand_ins)/delta_f - np.mean(rand_del)/delta_f

        results.append({
            'class': class_name, 'pred': pred_prob, 'base': base_prob,
            'audc': audc, 'auic': auic, 'faithfulness': faithfulness,
            'rand_faithfulness': rand_f,
            'del_curve': del_curve, 'ins_curve': ins_curve,
            'importances': importances
        })

        print(f"  {class_name:12s}: Pred={pred_prob:.3f}, Base={base_prob:.3f}, "
              f"AUDC={audc:.4f}, AUIC={auic:.4f}, F={faithfulness:+.4f} "
              f"(random: {rand_f:+.4f})")

    if skipped:
        print(f"\n  Skipped {len(skipped)} classes:")
        for name, reason in skipped:
            print(f"    {name}: {reason}")

    if results:
        mean_f = np.mean([r['faithfulness'] for r in results])
        mean_rf = np.mean([r['rand_faithfulness'] for r in results])
        std_f = np.std([r['faithfulness'] for r in results])
        print(f"\n  Mean Faithfulness: {mean_f:.4f} +/- {std_f:.4f} (over {len(results)} valid classes)")
        print(f"  Mean Random:       {mean_rf:.4f}")
        print(f"  Improvement:       {mean_f - mean_rf:+.4f}")

        # Plot
        n_valid = len(results)
        ncols = min(5, n_valid)
        nrows = max(1, (n_valid + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, r in enumerate(results):
            ax = axes_flat[idx]
            x = np.arange(n_regions + 1)
            ax.plot(x, r['del_curve'], 'r-o', markersize=3, label='Deletion')
            ax.plot(x, r['ins_curve'], 'g-o', markersize=3, label='Insertion')
            ax.set_title(f"{r['class']}\nF={r['faithfulness']:+.3f}", fontsize=9)
            ax.set_xlabel("Regions removed/added")
            ax.set_ylabel("Confidence")
            ax.set_facecolor('white')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        for idx in range(len(results), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}protocol4.png', dpi=600, bbox_inches='tight')
        plt.show()

    return results

p4_results = protocol_4(model, test_dataset, device)


def protocol_5(model, test_dataset, device, n_regions=8,ref=best_config[0], vs=best_config[1], output_dir=output_directory):
    """
    Protocol 5: Gradient alignment diagnostic with statistical testing.
    Tests on MULTIPLE samples for statistical power.
    """

    os.makedirs(output_dir, exist_ok=True)

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=500, device=device, value_space=vs
    )

    # Test on multiple samples for statistical power
    n_test = min(50, len(test_dataset))
    correlations = []

    print(f"\n1. Gradient Alignment (testing {n_test} samples)...")
    for i in range(n_test):
        pc, label = test_dataset[i]
        pc = pc.numpy()
        label = label if isinstance(label, int) else label.item()

        # Get R-SHAP importances
        importances, _, _, regions, _ = explainer.explain(pc, label, verbose=False)

        # Get gradient magnitudes per region
        model.eval()
        pc_tensor = torch.FloatTensor(pc).unsqueeze(0).to(device)
        pc_tensor.requires_grad = True
        logits = model(pc_tensor)
        logits[0, label].backward()
        grads = pc_tensor.grad.squeeze(0).detach().cpu().numpy()
        grad_magnitude = np.linalg.norm(grads, axis=1)

        # Aggregate gradients per region
        grad_per_region = np.array([
            grad_magnitude[regions == r].mean() for r in range(n_regions)
        ])

        # Spearman correlation
        if np.std(importances) > 1e-8 and np.std(grad_per_region) > 1e-8:
            corr, pval = spearmanr(importances, grad_per_region)
            correlations.append((corr, pval))

    if correlations:
        mean_corr = np.mean([c[0] for c in correlations])
        mean_p = np.mean([c[1] for c in correlations])
        frac_positive = np.mean([c[0] > 0 for c in correlations])
        frac_sig = np.mean([c[1] < 0.05 for c in correlations])

        print(f"  Mean Spearman correlation: {mean_corr:.4f}")
        print(f"  Fraction positive: {frac_positive:.1%}")
        print(f"  Fraction significant (p<0.05): {frac_sig:.1%}")
    else:
        print("  All importances had zero variance")

    # 2. Ensemble Consistency
    print(f"\n2. Ensemble Consistency...")
    pc, label = test_dataset[0]
    pc = pc.numpy()
    label = label if isinstance(label, int) else label.item()

    ensemble_importances = []
    for seed in range(5):
        exp = RegionSHAP(
            model=model, reference_mechanism=ref,
            n_regions=n_regions, n_samples=500, device=device, value_space=vs
        )
        imp, _, _, _, _ = exp.explain(pc, label, verbose=False)
        ensemble_importances.append(imp)

    ensemble_var = np.mean(np.var(ensemble_importances, axis=0))
    print(f"  Ensemble variance: {ensemble_var:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if correlations:
        axes[0].hist([c[0] for c in correlations], bins=20, edgecolor='black')
        axes[0].axvline(mean_corr, color='r', linestyle='--', label=f'Mean={mean_corr:.3f}')
        axes[0].set_facecolor('white')
        axes[0].set_xlabel("Spearman Correlation")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Gradient-SHAP Correlation Distribution")
        axes[0].legend()

    ens = np.array(ensemble_importances)
    for j in range(5):
        axes[1].plot(range(n_regions), ens[j], 'o-', alpha=0.5, label=f'Run {j+1}')
    axes[1].set_xlabel("Region")
    axes[1].set_facecolor('white')
    axes[1].set_ylabel("Importance")
    axes[1].set_title("Ensemble Consistency")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol5.png', dpi=600, bbox_inches='tight')
    plt.show()

    return correlations, ensemble_importances

p5_corr, p5_ens = protocol_5(model, test_dataset, device)


def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def protocol_6(model, test_dataset, device, n_regions=8,ref=best_config[0], vs=best_config[1],output_dir=output_directory):
    """
    Protocol 6: Rotation equivariance with model invariance baseline.
    """

    os.makedirs(output_dir, exist_ok=True)

    sample_pc, label = test_dataset[0]
    sample_pc = sample_pc.numpy()
    label = label if isinstance(label, int) else label.item()

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=500, device=device, value_space=vs
    )

    # Step 1: Check MODEL rotation invariance
    print("\n  Step 1: Checking model rotation invariance...")
    model.eval()
    with torch.no_grad():
        pc_tensor = torch.FloatTensor(sample_pc).unsqueeze(0).to(device)
        orig_logits = model(pc_tensor).squeeze(0).cpu().numpy()

    model_errors = []
    rng = np.random.RandomState(42)
    rotations = []
    for t in range(5):
        axis = rng.randn(3)
        theta = rng.uniform(0, 2 * np.pi)
        R = rotation_matrix(axis, theta)
        rotations.append(R)

        rotated_pc = (R @ sample_pc.T).T
        with torch.no_grad():
            rot_tensor = torch.FloatTensor(rotated_pc).unsqueeze(0).to(device)
            rot_logits = model(rot_tensor).squeeze(0).cpu().numpy()

        model_err = np.linalg.norm(orig_logits - rot_logits)
        model_errors.append(model_err)
        print(f"    Rotation {t+1}: model logit error = {model_err:.4f}")

    mean_model_err = np.mean(model_errors)
    print(f"  Mean model error: {mean_model_err:.4f}")

    # Step 2: Measure R-SHAP equivariance error
    print("  Step 2: Computing R-SHAP equivariance error...")
    orig_imp, _, _, _, _ = explainer.explain(sample_pc, label, verbose=False)

    rshap_errors = []
    for t, R in enumerate(rotations):
        rotated_pc = (R @ sample_pc.T).T
        rot_imp, _, _, _, _ = explainer.explain(rotated_pc, label, verbose=False)
        rshap_err = np.linalg.norm(rot_imp - orig_imp)
        rshap_errors.append(rshap_err)
        print(f"    Rotation {t+1}: R-SHAP error = {rshap_err:.4f} (model error = {model_errors[t]:.4f})")

    mean_rshap_err = np.mean(rshap_errors)
    print(f"\n  Mean R-SHAP error:  {mean_rshap_err:.4f}")
    print(f"  Mean model error:   {mean_model_err:.4f}")


    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, model_errors, w, label='Model Logit Error', color='blue', alpha=0.7)
    ax.bar(x + w/2, rshap_errors, w, label='R-SHAP Error', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_facecolor('white')
    ax.set_xticklabels([f'Rot {t+1}' for t in range(5)])
    ax.set_ylabel('L2 Error')
    ax.set_title('Rotation Equivariance: Model vs R-SHAP Error')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol6.png', dpi=600, bbox_inches='tight')
    plt.show()

    return model_errors, rshap_errors

p6_model_errors, p6_errors = protocol_6(model, test_dataset, device)


def protocol_7(model, test_dataset, device, n_regions=8,ref=best_config[0], vs=best_config[1], output_dir=output_directory):
    """
    Protocol 7: Per class with region partition.
    """

    os.makedirs(output_dir, exist_ok=True)

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=1000, device=device, value_space=vs
    )

    n_classes = len(test_dataset.classes)
    cmap_regions = plt.cm.get_cmap('tab10', n_regions)

    for class_idx in range(n_classes):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                pc = pc.numpy()
                class_name = test_dataset.classes[class_idx]
                break

        importances, baseline, prediction, regions, _ = explainer.explain(
            pc, class_idx, verbose=False
        )

        fig = plt.figure(figsize=(12, 4.5))
        fig.suptitle(f'{class_name}', fontsize=13, fontweight='bold', y=1.02)

        gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.05, right=0.96, bottom=0.08, top=0.88)

        # --- Plot 1: Region Partition ---
        ax1 = fig.add_subplot(gs[0], projection='3d')
        for r in range(n_regions):
            mask = regions == r
            ax1.scatter(pc[mask, 0], pc[mask, 1], pc[mask, 2],
                       c=[cmap_regions(r)], s=3, alpha=0.7, label=f'R{r}')
        ax1.set_xlabel('X', fontsize=8, labelpad=-2)
        ax1.set_ylabel('Y', fontsize=8, labelpad=-2)
        ax1.set_zlabel('Z', fontsize=8, labelpad=-2)
        ax1.set_title('Regions', fontsize=10, pad=2)
        ax1.set_facecolor('white')
        ax1.tick_params(axis='both', labelsize=6)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(n_regions, 8), fontsize=6, markerscale=2,
                  handletextpad=0.2, columnspacing=0.5, frameon=True)

        # --- Plot 2: Region Importances Bar Chart ---
        ax2 = fig.add_subplot(gs[1])
        bar_colors = [cmap_regions(r) for r in range(n_regions)]
        ax2.barh(range(n_regions), importances, color=bar_colors, alpha=0.8,
                edgecolor='black', linewidth=0.3)
        ax2.set_xlabel('Shapley Value', fontsize=9)
        ax2.set_ylabel('Region', fontsize=9)
        ax2.set_yticks(range(n_regions))
        ax2.set_yticklabels([f'R{r}' for r in range(n_regions)], fontsize=7)
        ax2.set_title(f'Region Importances (Std={np.std(importances):.3f})', fontsize=10, pad=2)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_facecolor('white')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.tick_params(axis='x', labelsize=7)

        plt.savefig(f'{output_dir}protocol7_{class_name}.png', dpi=600, bbox_inches='tight')
        # plt.savefig(f'{output_dir}protocol7_{class_name}.eps', format='eps', bbox_inches='tight')
        plt.show()


protocol_7(model, test_dataset, device)


def protocol_8(model, test_dataset, device, n_regions=8,ref=best_config[0], vs=best_config[1],output_dir=output_directory):

    os.makedirs(output_dir, exist_ok=True)

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=1000, device=device, value_space=vs
    )

    stats = []
    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                pc = pc.numpy()
                class_name = test_dataset.classes[class_idx]
                break

        importances, baseline, prediction, regions, _ = explainer.explain(
            pc, class_idx, verbose=False
        )

        stats.append({
            'class': class_name,
            'min': np.min(importances),
            'max': np.max(importances),
            'mean': np.mean(importances),
            'std': np.std(importances),
            'range': np.max(importances) - np.min(importances),
            'positive_ratio': np.mean(importances > 0),
            'importances': importances
        })

        print(f"  {class_name:12s}: Min={importances.min():+.4f}, Max={importances.max():+.4f}, "
              f"Std={np.std(importances):.4f}, Range={np.max(importances)-np.min(importances):.4f}")

    # Summary table
    df = pd.DataFrame([{k: v for k, v in s.items() if k != 'importances'} for s in stats])
    print(f"\n{df.to_string(index=False)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Importance heatmap
    imp_matrix = np.array([s['importances'] for s in stats])
    im = axes[0].imshow(imp_matrix, aspect='auto', cmap='RdYlGn')
    axes[0].set_yticks(range(len(stats)))
    axes[0].set_yticklabels([s['class'] for s in stats], fontsize=8)
    axes[0].set_facecolor('white')
    axes[0].set_xlabel("Region")
    axes[0].set_title("Region Importances by Class")
    plt.colorbar(im, ax=axes[0])

    # Std comparison
    axes[1].barh([s['class'] for s in stats], [s['std'] for s in stats])
    axes[1].set_facecolor('white')
    axes[1].set_xlabel("Importance Std")
    axes[1].set_title("Importance Variation by Class")

    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol8.png', dpi=600, bbox_inches='tight')
    plt.show()

    df.to_csv(f'{output_dir}protocol8.csv', index=False)
    return stats

p8_stats = protocol_8(model, test_dataset, device)


def protocol_10_reference_comparison(model, test_dataset, device, n_regions=8,output_dir=output_directory):
    """
    Protocol 10: Reference Mechanism Comparison.
    GIR is the primary mechanism. Zero/Mean/Noise are ablation comparisons.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 10: Reference Mechanism Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    refs = ['gir', 'zero', 'mean', 'noise']

    samples = []
    for class_idx in range(min(5, len(test_dataset.classes))):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                samples.append((pc.numpy(), class_idx, test_dataset.classes[class_idx]))
                break

    all_results = {ref: [] for ref in refs}

    for ref in refs:
        for pc, class_idx, class_name in samples:
            explainer = RegionSHAP(
                model=model, reference_mechanism=ref,
                n_regions=n_regions, n_samples=500, device=device, value_space='logit'
            )
            imp, base, pred, regions, _ = explainer.explain(pc, class_idx, verbose=False)
            all_results[ref].append({
                'class': class_name, 'importances': imp,
                'std': np.std(imp), 'range': np.max(imp) - np.min(imp),
                'delta': pred - base
            })

        avg_std = np.mean([r['std'] for r in all_results[ref]])
        avg_range = np.mean([r['range'] for r in all_results[ref]])
        avg_delta = np.mean([r['delta'] for r in all_results[ref]])
        marker = " [PRIMARY]" if ref == 'gir' else ""
        print(f"  {ref:6s}: Avg Std={avg_std:.4f}, Avg Range={avg_range:.4f}, Avg Delta={avg_delta:.4f}{marker}")

    # Check if zero == mean (expected for centered point clouds)
    zero_stds = [r['std'] for r in all_results['zero']]
    mean_stds = [r['std'] for r in all_results['mean']]

    # Highlight GIR
    gir_std = np.mean([r['std'] for r in all_results['gir']])
    zero_std = np.mean([r['std'] for r in all_results['zero']])
    print(f"\n  GIR differentiation (Avg Std={gir_std:.4f}) vs Zero (Avg Std={zero_std:.4f})")
    if gir_std > zero_std:
        print(f"  GIR produces MORE differentiated importances than Zero ({gir_std/zero_std:.1f}x)")
    else:
        print(f"  Zero produces more differentiated importances than GIR ({zero_std/gir_std:.1f}x)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stds = {ref: np.mean([r['std'] for r in all_results[ref]]) for ref in refs}
    ranges_val = {ref: np.mean([r['range'] for r in all_results[ref]]) for ref in refs}

    colors = ['green', 'red', 'blue', 'orange']
    edge_colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange']

    bars1 = axes[0].bar(refs, [stds[r] for r in refs], color=colors, edgecolor=edge_colors, linewidth=1.5)
    axes[0].set_ylabel("Avg Importance Std")
    axes[0].set_title("Signal Strength by Reference Mechanism")
    axes[0].set_facecolor('white')
    # Mark primary
    bars1[0].set_hatch('//')
    axes[0].legend([bars1[0]], ['Primary (GIR)'], fontsize=8)

    bars2 = axes[1].bar(refs, [ranges_val[r] for r in refs], color=colors, edgecolor=edge_colors, linewidth=1.5)
    axes[1].set_ylabel("Avg Importance Range")
    axes[1].set_facecolor('white')
    axes[1].set_title("Importance Differentiation by Reference")
    bars2[0].set_hatch('//')

    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol10_refs.png', dpi=600, bbox_inches='tight')
    plt.show()

    return all_results

p10_results = protocol_10_reference_comparison(model, test_dataset, device)


def protocol_11_critical_points(model, test_dataset, device, n_regions=8,output_dir=output_directory):

    os.makedirs(output_dir, exist_ok=True)

    all_distributions = []

    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                pc = pc.numpy()
                class_name = test_dataset.classes[class_idx]
                break

        # Get pre-maxpool features
        model.eval()
        with torch.no_grad():
            pc_tensor = torch.FloatTensor(pc).unsqueeze(0).to(device)
            _, global_feat, _ = model(pc_tensor, return_features=True)

            # Get features before max-pool (1024 channels)
            x = pc_tensor.transpose(2, 1)
            if model.use_tnet:
                trans = model.tnet1(x)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans)
                x = x.transpose(2, 1)
            x = F.relu(model.bn1(model.conv1(x)))
            x = F.relu(model.bn2(model.conv2(x)))
            if model.use_tnet:
                trans_feat = model.tnet2(x)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2, 1)
            x = F.relu(model.bn3(model.conv3(x)))
            x = F.relu(model.bn4(model.conv4(x)))
            x = model.bn5(model.conv5(x))  # (1, 1024, N)

            # Find critical points (argmax per channel)
            critical_indices = torch.argmax(x.squeeze(0), dim=1).cpu().numpy()  # (1024,)

        # Segment regions
        explainer = RegionSHAP(model=model, n_regions=n_regions, device=device)
        regions = explainer.segment_regions(pc)

        # Count critical points per region
        unique_critical = np.unique(critical_indices)
        critical_per_region = np.zeros(n_regions)
        for cp in critical_indices:
            critical_per_region[regions[cp]] += 1

        # Normalize
        critical_frac = critical_per_region / critical_per_region.sum()
        region_sizes = np.array([(regions == r).sum() for r in range(n_regions)])
        region_frac = region_sizes / region_sizes.sum()

        # Critical point concentration (entropy-based)
        from scipy.stats import entropy
        cp_entropy = entropy(critical_frac + 1e-10)
        uniform_entropy = np.log(n_regions)
        concentration = 1 - cp_entropy / uniform_entropy

        all_distributions.append({
            'class': class_name,
            'n_unique_critical': len(unique_critical),
            'critical_per_region': critical_per_region,
            'critical_frac': critical_frac,
            'region_frac': region_frac,
            'concentration': concentration
        })

        cp_str = " ".join([f"{f:.2f}" for f in critical_frac])
        print(f"  {class_name:12s}: {len(unique_critical):3d} unique critical points, "
              f"concentration={concentration:.3f}, dist=[{cp_str}]")

    mean_conc = np.mean([d['concentration'] for d in all_distributions])
    mean_unique = np.mean([d['n_unique_critical'] for d in all_distributions])

    print(f"\n  Mean unique critical points: {mean_unique:.0f} / 1024 channels")
    print(f"  Mean concentration: {mean_conc:.3f} (0=uniform, 1=single region)")

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for idx, d in enumerate(all_distributions):
        ax = axes[idx // 5, idx % 5]
        x = np.arange(n_regions)
        w = 0.35
        ax.bar(x - w/2, d['critical_frac'], w, label='Critical pts', color='red', alpha=0.7)
        ax.bar(x + w/2, d['region_frac'], w, label='Region size', color='blue', alpha=0.5)
        ax.set_title(f'{d["class"]}\nconc={d["concentration"]:.2f}', fontsize=9)
        ax.set_xlabel("Region")
        ax.set_facecolor('white')
        if idx % 5 == 0:
            ax.set_ylabel("Fraction")
        ax.legend(fontsize=6)

    plt.suptitle("Critical Point Distribution vs Region Size", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol11_critical.png', dpi=600, bbox_inches='tight')
    plt.show()

    return all_distributions

p11_results = protocol_11_critical_points(model, test_dataset, device)


def protocol_12(model, test_dataset, device, n_regions=8,ref=best_config[0], vs=best_config[1],output_dir=output_directory):
    """
    Protocol 9: Ablation study.
    """

    os.makedirs(output_dir, exist_ok=True)

    explainer = RegionSHAP(
        model=model, reference_mechanism=ref,
        n_regions=n_regions, n_samples=1000, device=device, value_space=vs
    )

    results = []
    for class_idx in range(len(test_dataset.classes)):
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            if label == class_idx:
                pc, _ = test_dataset[i]
                pc = pc.numpy()
                class_name = test_dataset.classes[class_idx]
                break

        importances, baseline, prediction, regions, _ = explainer.explain(
            pc, class_idx, verbose=False
        )

        # Use the SAME evaluate_model (logit or prob) as SHAP computation
        pred_val = explainer.evaluate_model(pc, class_idx)

        # Remove most important region
        most_imp = np.argmax(importances)
        coalition_imp = np.ones(n_regions)
        coalition_imp[most_imp] = 0
        drop_imp = pred_val - explainer.evaluate_model(
            explainer.perturb_coalition(pc, coalition_imp, regions), class_idx
        )

        # Remove least important region
        least_imp = np.argmin(importances)
        coalition_least = np.ones(n_regions)
        coalition_least[least_imp] = 0
        drop_least = pred_val - explainer.evaluate_model(
            explainer.perturb_coalition(pc, coalition_least, regions), class_idx
        )

        # Remove random region (avg of 3)
        rng = np.random.RandomState(0)
        rand_drops = []
        for _ in range(3):
            rand_r = rng.randint(n_regions)
            coal_r = np.ones(n_regions)
            coal_r[rand_r] = 0
            rd = pred_val - explainer.evaluate_model(
                explainer.perturb_coalition(pc, coal_r, regions), class_idx
            )
            rand_drops.append(rd)
        drop_rand = np.mean(rand_drops)

        results.append({
            'class': class_name,
            'most_imp_region': most_imp,
            'most_imp_phi': importances[most_imp],
            'drop_most': drop_imp,
            'least_imp_region': least_imp,
            'least_imp_phi': importances[least_imp],
            'drop_least': drop_least,
            'drop_random': drop_rand
        })

        print(f"  {class_name:12s}: Most imp R{most_imp} (phi={importances[most_imp]:+.4f}) -> Drop: {drop_imp:.4f} "
              f"| Least R{least_imp} (phi={importances[least_imp]:+.4f}) -> Drop: {drop_least:.4f} "
              f"| Random: {drop_rand:.4f}")

    mean_drop_most = np.mean([r['drop_most'] for r in results])
    mean_drop_least = np.mean([r['drop_least'] for r in results])
    mean_drop_rand = np.mean([r['drop_random'] for r in results])

    print(f"\n  Value space: {vs} (same space as SHAP computation)")
    print(f"  Mean drop (most important):  {mean_drop_most:.4f}")
    print(f"  Mean drop (least important): {mean_drop_least:.4f}")
    print(f"  Mean drop (random):          {mean_drop_rand:.4f}")
    if abs(mean_drop_rand) > 1e-6:
        print(f"  Ratio (most/random):         {mean_drop_most/mean_drop_rand:.2f}x")
    if abs(mean_drop_least) > 1e-6:
        print(f"  Ratio (most/least):          {mean_drop_most/mean_drop_least:.2f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results))
    w = 0.25
    ax.bar(x - w, [r['drop_most'] for r in results], w, label='Most Important', color='red', alpha=0.7)
    ax.bar(x, [r['drop_random'] for r in results], w, label='Random', color='gray', alpha=0.7)
    ax.bar(x + w, [r['drop_least'] for r in results], w, label='Least Important', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([r['class'] for r in results], rotation=45, ha='right', fontsize=8)
    ax.set_facecolor('white')
    ax.set_ylabel(f"Drop ({vs} space)")
    ax.set_title(f"Ablation: Remove Most vs Least vs Random Region ({vs} space)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}protocol12.png', dpi=600, bbox_inches='tight')
    plt.show()

    return results

p12_results = protocol_12(model, test_dataset, device)


