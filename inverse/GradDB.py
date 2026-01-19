import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from main_model.GridTransformer import GridTransformer
from scripts.mechanism_expand import expand_mechanism, forward_kinematics_torch

def load_model(model_path, device='cpu'):
    dh_params = torch.tensor([[89.159, 0.0,    0.0, torch.pi/2],
                              [   0.0, 0.0, -425.0, 0.0],
                              [   0.0, 0.0,-392.25, 0.0],
                              [109.15, 0.0,    0.0, torch.pi/2],
                              [ 94.65, 0.0,    0.0, -torch.pi/2],
                              [  82.3, 0.0,    0.0, 0.0]], device=device)
    tool_offset = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], device=device)
    model_params = {
        "mechanism_num_inputs": 6 * 4 + 1,
        "para_num_inputs": 6,
        "mechanism_input_dim": 3,
        "para_input_dim": 1,
        "mechanism_hidden_dims": [128, 128],
        "para_hidden_dims": [128, 128],
        "embed_out_dim": 128,
        "normalization": "batchnorm",
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout_rate": 0.3,
        "encoder_dim": 256,
        "m_pos_enc_max_len": 6 * 4 + 1,
        "p_pos_enc_max_len": 6,
        "pos_enc_init_method": "normal",
        "pos_enc_scale": 1.0,
        "use_mechanism_pos_enc": True,
        "use_para_pos_enc": True,
        "dh_params": dh_params,
        "tool_offset": tool_offset,
        "mechanism_offsets": [-2.0, -1.0, 1.0, 2.0]
    }

    device = torch.device(device)
    net = GridTransformer(**model_params).to(device)
    
    net.load_state_dict(torch.load(model_path)['state_dict'])
    net.requires_grad_(False)
    net.eval()
    
    return net

def inverse_kinematics_with_initial_angles(model, target_pos, initial_angles, device='cpu', 
                                          max_iters=500, initial_lr=1e-3,
                                          lambda_start=0.1, lambda_end=0.001, error_threshold=9e-3):
    if len(target_pos.shape) == 1:
        target_pos = torch.tensor(target_pos, device=device).unsqueeze(0)
    else:
        target_pos = torch.tensor(target_pos, device=device)
    
    if isinstance(initial_angles, np.ndarray):
        if len(initial_angles.shape) == 1:
            initial_angles = torch.tensor(initial_angles, device=device).unsqueeze(0)
        else:
            initial_angles = torch.tensor(initial_angles, device=device)
    else:
        if len(initial_angles.shape) == 1:
            initial_angles = initial_angles.unsqueeze(0).to(device)
        else:
            initial_angles = initial_angles.to(device)

    q_init = initial_angles.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([q_init], lr=initial_lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=initial_lr/100)

    dh_params = model.dh_params
    tool_offset = model.tool_offset
    mechanism_offsets = model.mechanism_offsets

    best_loss = float('inf')
    best_angles = q_init.clone().detach()
    no_improve_count = 0

    start_time = time.time()
    last_report_time = start_time
    report_interval = 5

    loss_history = []
    pos_loss_history = []
    angle_loss_history = []
    iter_history = []

    for iter in range(max_iters):
        optimizer.zero_grad()

        expand_mechanism_data, _ = expand_mechanism(q_init, dh_params, tool_offset, mechanism_offsets)

        para_data = q_init.unsqueeze(-1)

        pred_pos = model(expand_mechanism_data, para_data)

        pos_loss = torch.sqrt(torch.mean(torch.sum((pred_pos + expand_mechanism_data[:, 0, :] - target_pos) ** 2, dim=1)))

        angle_loss = torch.mean(torch.sum((q_init - initial_angles) ** 2, dim=1))

        progress = iter / max_iters
        lambdaL = lambda_start * (1 - progress) + lambda_end * progress

        loss = (1 - lambdaL) * pos_loss + lambdaL * angle_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        pos_loss_history.append(pos_loss.item())
        angle_loss_history.append(angle_loss.item())
        iter_history.append(iter)

        current_time = time.time()
        if current_time - last_report_time > report_interval or iter == 0 or iter == max_iters-1:
            elapsed = current_time - start_time
            print(f"Iteration {iter}/{max_iters}, Loss: {loss.item():.8f}, Position Loss: {pos_loss.item():.8f}, "
                  f"Angle Loss: {angle_loss.item():.8f}, Learning Rate: {scheduler.get_last_lr()[0]:.8f}, "
                  f"Lambda: {lambdaL:.5f}, Time Elapsed: {elapsed:.2f}s")
            last_report_time = current_time

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_angles = q_init.clone().detach()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if pos_loss.item() < error_threshold:
            print(f"Early convergence at iteration {iter+1}/{max_iters}, Loss: {loss.item():.8f}")
            break

    optimized_angles = best_angles
    
    optimized_angles_deg = optimized_angles

    expand_mechanism_data, _ = expand_mechanism(optimized_angles, dh_params, tool_offset, mechanism_offsets)
    para_data = optimized_angles.unsqueeze(-1)
    actual_position = model(expand_mechanism_data, para_data) + expand_mechanism_data[:, 0, :]

    position_error = torch.abs(actual_position - target_pos)

    final_error_mean = torch.mean(position_error).item()
    print(f"Optimization complete. Final average position error: {final_error_mean:.8f}")

    loss_data = {
        'iterations': iter_history,
        'total_loss': loss_history,
        'position_loss': pos_loss_history,
        'angle_loss': angle_loss_history
    }
    
    return optimized_angles_deg, actual_position, position_error, loss_data

def analyze_delta_theta(optimized_angles_deg, original_angles, output_dir):
    """Analyze delta theta data and plot histograms.
    
    Args:
        optimized_angles_deg: Optimized angles (degrees)
        original_angles: Original input angles (degrees)
        output_dir: Output directory path
    """
    # Calculate delta theta
    delta_theta = optimized_angles_deg - original_angles
    
    # Create directory for saving histograms
    delta_dir = os.path.join(output_dir, "delta_theta")
    os.makedirs(delta_dir, exist_ok=True)
    
    # Convert to NumPy array for processing
    if isinstance(delta_theta, torch.Tensor):
        delta_theta = delta_theta.cpu().numpy()
    
    # Calculate statistics
    stats = {
        "max": np.max(delta_theta, axis=0),
        "min": np.min(delta_theta, axis=0),
        "mean": np.mean(delta_theta, axis=0),
        "variance": np.var(delta_theta, axis=0),
        "std": np.std(delta_theta, axis=0)
    }
    
    # Calculate absolute delta statistics
    abs_delta = np.abs(delta_theta)
    abs_stats = {
        "abs_max": np.max(abs_delta, axis=0),
        "abs_min": np.min(abs_delta, axis=0),
        "abs_mean": np.mean(abs_delta, axis=0),
        "abs_variance": np.var(abs_delta, axis=0),
        "abs_std": np.std(abs_delta, axis=0)
    }
    
    # Print statistics
    print("\nDelta Theta Statistics:")
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    
    for dim in range(delta_theta.shape[1]):
        joint_name = joint_names[dim] if dim < len(joint_names) else f"Joint {dim+1}"
        print(f"\n{joint_name}:")
        print(f"  Max: {stats['max'][dim]:.6f}")
        print(f"  Min: {stats['min'][dim]:.6f}")
        print(f"  Mean: {stats['mean'][dim]:.6f}")
        print(f"  Variance: {stats['variance'][dim]:.6f}")
        print(f"  Std Dev: {stats['std'][dim]:.6f}")
        
        # Absolute value statistics
        print(f"\n{joint_name} (Absolute Values):")
        print(f"  Abs Max: {abs_stats['abs_max'][dim]:.6f}")
        print(f"  Abs Min: {abs_stats['abs_min'][dim]:.6f}")
        print(f"  Abs Mean: {abs_stats['abs_mean'][dim]:.6f}")
        print(f"  Abs Variance: {abs_stats['abs_variance'][dim]:.6f}")
        print(f"  Abs Std Dev: {abs_stats['abs_std'][dim]:.6f}")
    
    # Plot histograms for each dimension (regular and absolute values)
    for dim in range(delta_theta.shape[1]):
        joint_name = joint_names[dim] if dim < len(joint_names) else f"joint_{dim+1}"
        
        # Plot regular delta theta histogram
        plt.figure(figsize=(8, 6))
        plt.hist(delta_theta[:, dim], bins=50, color="#3498db", alpha=0.7, 
                edgecolor="black")
        plt.title(f"Delta Theta - {joint_name}")
        plt.xlabel("Value (degrees)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(delta_dir, f"delta_theta_dim_{dim}.png"))
        plt.close()
        
        # Plot absolute delta theta histogram
        plt.figure(figsize=(8, 6))
        plt.hist(abs_delta[:, dim], bins=50, color="#e74c3c", alpha=0.7, 
                edgecolor="black")
        plt.title(f"Absolute Delta Theta - {joint_name}")
        plt.xlabel("Value (degrees)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(delta_dir, f"abs_delta_theta_dim_{dim}.png"))
        plt.close()
    
    # Plot enhanced histograms (with KDE curve)
    for dim in range(delta_theta.shape[1]):
        joint_name = joint_names[dim] if dim < len(joint_names) else f"joint_{dim+1}"
        plot_enhanced_histogram(
            delta_theta[:, dim],
            f"Delta Theta - {joint_name}",
            os.path.join(delta_dir, f"enhanced_delta_theta_dim_{dim}.png")
        )
        
        plot_enhanced_histogram(
            abs_delta[:, dim],
            f"Absolute Delta Theta - {joint_name}",
            os.path.join(delta_dir, f"enhanced_abs_delta_theta_dim_{dim}.png"),
            is_absolute=True
        )
    
    return delta_theta, abs_delta

def plot_enhanced_histogram(data, title, save_path, is_absolute=False):
    """Plot enhanced histogram with KDE curve and statistics.
    
    Args:
        data: Data array
        title: Chart title
        save_path: Save path
        is_absolute: Whether the data is absolute values
    """
    from scipy import stats
    
    # Calculate statistics
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Determine x-axis range
    if is_absolute:
        # For absolute values, x-axis starts at 0
        x_min, x_max = 0, np.max(data) * 1.1
    else:
        # For original values, dynamically adjust x-axis range
        sigma2_low = mean - 2 * std_dev
        sigma2_high = mean + 2 * std_dev
        data_min = np.min(data)
        data_max = np.max(data)
        plot_min = min(data_min, sigma2_low)
        plot_max = max(data_max, sigma2_high)
        range_extension = 0.1 * (sigma2_high - sigma2_low)
        x_min = plot_min - range_extension
        x_max = plot_max + range_extension
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Plot histogram with density instead of frequency
    plt.hist(data, bins=50, color="#3498db" if not is_absolute else "#2ecc71", alpha=0.7, 
            edgecolor="black", density=True)
    
    # Add KDE curve
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(x_min, x_max, 1000)
    kde_values = kde(x_range)
    plt.plot(x_range, kde_values, 'r-', linewidth=2, 
            label=f'KDE Variance: {np.var(data):.6f}')
    
    # Find KDE peak position
    peak_idx = np.argmax(kde_values)
    peak_x = x_range[peak_idx]
    peak_y = kde_values[peak_idx]
    
    # Plot KDE peak vertical line
    plt.plot([peak_x, peak_x], [0, peak_y], color='green', linestyle='--', linewidth=2, 
            label=f'Peak: {peak_x:.6f}')
    
    if is_absolute:
        # For absolute values, calculate percentiles
        top_90_percentile = np.percentile(data, 90)
        top_95_percentile = np.percentile(data, 95)
        top_99_percentile = np.percentile(data, 99)
        
        # Set x-axis range for visibility
        plt.xlim(0, min(x_max, top_99_percentile * 1.5))
        
        # Add statistics text
        stats_text = (f"Mean: {mean:.6f}\n"
                     f"Median: {np.median(data):.6f}\n"
                     f"Std Dev: {std_dev:.6f}\n"
                     f"90% Percentile: {top_90_percentile:.6f}\n"
                     f"95% Percentile: {top_95_percentile:.6f}")
        
        # Add threshold lines
        plt.axvline(x=0.1, color='purple', linestyle='--', linewidth=1, 
                   label='0.1° Threshold')
        plt.axvline(x=0.01, color='blue', linestyle='--', linewidth=1, 
                   label='0.01° Threshold')
    else:
        # Calculate 2sigma range
        sigma2_low = mean - 2 * std_dev
        sigma2_high = mean + 2 * std_dev
        
        # Get 2sigma boundary KDE curve heights
        sigma2_low_y = kde(sigma2_low)[0] if sigma2_low >= x_min else 0
        sigma2_high_y = kde(sigma2_high)[0] if sigma2_high <= x_max else 0
        
        # Plot 2sigma range vertical lines
        plt.plot([sigma2_low, sigma2_low], [0, sigma2_low_y], color='#f39c12', linestyle='--', linewidth=2)
        plt.plot([sigma2_high, sigma2_high], [0, sigma2_high_y], color='#f39c12', linestyle='--', linewidth=2)
        
        # Add fill color for 2sigma range
        mask = (x_range >= sigma2_low) & (x_range <= sigma2_high)
        sigma2_x = x_range[mask]
        sigma2_y = kde_values[mask]
        plt.fill_between(sigma2_x, 0, sigma2_y, color='#f39c12', alpha=0.3)
        
        # Add legend label
        plt.plot([], [], color='#f39c12', alpha=0.3, linewidth=10, 
               label=f'2σ Range: [{sigma2_low:.6f}, {sigma2_high:.6f}]')
        
        # Calculate 95% confidence interval
        confidence_interval = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
        
        # Add statistics text
        stats_text = (f"Mean: {mean:.6f}\n"
                     f"Std Dev: {std_dev:.6f}\n"
                     f"Variance: {np.var(data):.6f}\n"
                     f"95% CI: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
    
    # Add statistics text box
    plt.annotate(stats_text, xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                va="top", ha="left", fontsize=9)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=10)
    
    # Set title and labels
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# main processing function
def process_batch_data(model_path, input_csv, output_csv, device='cpu'):
    model = load_model(model_path, device)
    print(f"Model used: {model_path}")
    print(f"Test dataset: {input_csv}")

    # load data
    df = pd.read_csv(input_csv, index_col=False)
    angles = df[['1', '2', '3', '4', '5', '6']].values
    rad_theta = torch.deg2rad(torch.tensor(angles, device=device))
    cal_mechanism, _ = forward_kinematics_torch(rad_theta, model.dh_params, model.tool_offset)

    total_start_time = time.time()

    print(f"Starting batch optimization for {len(df)} samples...")
    optimized_angles_deg, actual_positions, position_errors, loss_data = inverse_kinematics_with_initial_angles(
        model, 
        cal_mechanism, 
        angles, 
        device, 
        max_iters=10000, 
        initial_lr=5e-3,
        lambda_start=0.1,
        lambda_end=0.001,
        error_threshold=9e-3
    )

    total_time = time.time() - total_start_time
    print(f"Batch optimization complete! Total time: {total_time:.2f}s, Average per sample: {total_time/len(df):.4f}s")

    optimized_angles_deg = optimized_angles_deg.cpu().numpy()
    actual_positions = actual_positions.cpu().numpy()
    position_errors = position_errors.cpu().numpy()

    mean_error = np.mean(position_errors, axis=0)
    max_error = np.max(position_errors, axis=0)
    print(f"Average position error: x={mean_error[0]:.6f}, y={mean_error[1]:.6f}, z={mean_error[2]:.6f}")
    print(f"Maximum position error: x={max_error[0]:.6f}, y={max_error[1]:.6f}, z={max_error[2]:.6f}")

    result_dict = {}

    for i, col in enumerate(['θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6']):
        result_dict[col] = df.iloc[:, i].values
    
    cal_mechanism = cal_mechanism.cpu().numpy()

    result_dict['x'] = cal_mechanism[:, 0]
    result_dict['y'] = cal_mechanism[:, 1]
    result_dict['z'] = cal_mechanism[:, 2]

    for i, col in enumerate(['θ1_optimized', 'θ2_optimized', 'θ3_optimized', 'θ4_optimized', 'θ5_optimized', 'θ6_optimized']):
        result_dict[col] = optimized_angles_deg[:, i]

    result_dict['x-real'] = actual_positions[:, 0]
    result_dict['y-real'] = actual_positions[:, 1]
    result_dict['z-real'] = actual_positions[:, 2]
    result_dict['x-error'] = position_errors[:, 0]
    result_dict['y-error'] = position_errors[:, 1]
    result_dict['z-error'] = position_errors[:, 2]

    euclidean_error = np.sqrt(np.mean(position_errors**2, axis=1))
    result_dict['euclidean_error'] = euclidean_error

    result_df = pd.DataFrame(result_dict)

    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    delta_theta, abs_delta = analyze_delta_theta(optimized_angles_deg, angles, output_dir)
    print(f"Delta theta analysis charts saved to: {os.path.join(output_dir, 'delta_theta')}")

    plt.figure(figsize=(14, 7))

    plt.plot(loss_data['iterations'], loss_data['total_loss'], 'b-', label='Total Loss')
    plt.plot(loss_data['iterations'], loss_data['position_loss'], 'r-', label='Position Loss')
    plt.plot(loss_data['iterations'], loss_data['angle_loss'], 'g-', label='Angle Loss')

    plt.title('Loss Curves during Gradient Descent Optimization')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    img_dir = os.path.dirname(output_csv)
    img_path = os.path.join(img_dir, 'loss_curve_grad_descent.png')
    plt.savefig(img_path, dpi=300)
    print(f"Loss curve saved to: {img_path}")

    print("\n--- Summary Statistics ---")
    print(f"Total samples: {len(df)}")
    print(f"Euclidean distance error mean: {np.mean(euclidean_error):.6f}")
    print(f"Euclidean distance error median: {np.median(euclidean_error):.6f}")
    print(f"Euclidean distance error maximum: {np.max(euclidean_error):.6f}")
    print(f"Euclidean distance error minimum: {np.min(euclidean_error):.6f}")
    print(f"Percentage of samples with euclidean error < 0.01: {np.sum(euclidean_error < 0.01)/len(euclidean_error)*100:.2f}%")
    print(f"Percentage of samples with euclidean error < 0.001: {np.sum(euclidean_error < 0.001)/len(euclidean_error)*100:.2f}%")

    return {
        'optimized_angles': optimized_angles_deg,
        'actual_positions': actual_positions,
        'position_errors': position_errors,
        'loss_data': loss_data,
        'euclidean_error': euclidean_error
    }


if __name__ == "__main__":

    model_path = "../checkpoints/checkpoint.ckpt"

    input_csv = "../data/merge2copy.csv"

    output_csv = "../data/GDResult.csv"
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    results = process_batch_data(model_path, input_csv, output_csv, device)
    print("All processing complete!")
