#!/usr/bin/env python
r"""
A simplified version of GFlowNet training on the HyperGrid environment, focusing on the core concepts.
This script implements Trajectory Balance (TB) training with minimal features to aid understanding.

Example usage:
python train_hypergrid_simple.py --ndim 2 --height 8 --epsilon 0.1

Key differences from the full version:
- Only implements TB loss
- No replay buffer
- No wandb integration
- Simpler architecture with shared trunks
- Basic command line options
"""

import argparse
from typing import cast

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP
from gfn.utils.training import validate
from GrowingTriangleSampler import GrowingTriangleSampler
from gfn.modules import GFNModule
from gfn.gflownet import FMGFlowNet
import os

def visualize_trajectories(trajectories, env, it, sampler):

    # Only visualize when the environment is 2D.
    if env.ndim != 2:
        return

    # Create a new figure.
    plt.figure()
    for traj in trajectories:
        # Assume each trajectory has an attribute 'states'
        # which is a DiscreteStates instance with a 'tensor' attribute.
        if hasattr(traj.states, "tensor"):
            # reduce dimensionality to 2D
            points = traj.states.tensor.cpu().numpy()[:-1].reshape(-1, 2)
        else:
            points = traj.states.cpu().numpy()[:-1].reshape(-1, 2)
        # Plot the trajectory as a line with markers.
        plt.plot(points[:, 0], points[:, 1], marker='o', linestyle='-', alpha=0.6)

    plt.title(f"Trajectories at Iteration {it + 1} using {sampler} sampler")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.xlim(-0.5, env.height - 0.5)
    plt.ylim(-0.5, env.height - 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.pause(0.01)
    plt.close()
    
def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Setup the Environment.
    env = HyperGrid(ndim=args.ndim, height=args.height, device=device, calculate_all_states=True, calculate_partition=True, R0=0.00001, R1 = 1, R2=3)
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=module_PF.trunk,
    )
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)
    if args.sampler != "normal":
        sampler = GrowingTriangleSampler(pf_estimator,n_iterations=args.n_iterations, topological_type=args.sampler)

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})
    l1_distances = []
    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    # Get distributions and find global min/max for consistent color scaling.
    true_dist = env.true_dist_pmf.reshape(env.height, env.height).cpu().numpy()
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        # visualize trajectories on the grid 
        # visualize_trajectories(trajectories, env, it, args.sampler)
        loss.backward()
        optimizer.step()
        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            print(f"Iter {it + 1}: L1 distance {validation_info['l1_dist']:.8f}")
            l1_distances.append(validation_info["l1_dist"])
            plot_results(
                env,
                gflownet,
                true_dist,
                l1_distances,
                sampler=args.sampler,
            )
        pbar.set_postfix({"loss": loss.item()})

def plot_results(env, gflownet,true_dist, l1_distances, sampler="normal"):
    # Create directory for saving plots.
    save_dir = f"test_{sampler}_{env.height}_{env.ndim}"
    os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 1 row and 3 columns.
    fig = plt.figure(constrained_layout=True, figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    learned_dist = get_exact_P_T(env, gflownet).reshape(env.height, env.height).numpy()

    # Ensure consistent orientation by transposing.
    true_dist = true_dist.T
    learned_dist = learned_dist.T

    vmin = min(true_dist.min(), learned_dist.min())
    vmax = max(true_dist.max(), learned_dist.max())

    # True reward distribution.
    im1 = ax1.imshow(
        true_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("True Distribution")

    # Learned reward distribution.
    im2 = ax2.imshow(
        learned_dist,
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Learned Distribution")

    # Add a single colorbar for both distribution plots.
    fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", label="Probability")

    # Plot L1 distances over validation steps in the third subplot.
    ax3.plot(l1_distances, marker="o", linestyle="-")
    ax3.set_title("L1 Distances Over Validation Steps")
    ax3.set_xlabel("Validation Step")
    ax3.set_ylabel("L1 Distance")
    
    # Save the figure to the specified directory with a unique file name.
    filename = os.path.join(save_dir, f"results_{len(l1_distances):03d}.png")
    plt.savefig(filename)
    plt.close(fig)

def get_exact_P_T(env: HyperGrid, gflownet) -> torch.Tensor:
    r"""Evaluates the exact terminating state distribution P_T for HyperGrid.

    For each state s', the terminating state probability is computed as:

    .. math::
        P_T(s') = u(s') P_F(s_f | s')

    where u(s') satisfies the recursion:

    .. math::
        u(s') = \sum_{s \in \text{Par}(s')} u(s) P_F(s' | s)

    with the base case u(s_0) = 1.

    Args:
        env: The HyperGrid environment
        gflownet: The GFlowNet model

    Returns:
        The exact terminating state distribution as a tensor
    """
    if env.ndim != 2:
        raise ValueError("plotting is only supported for 2D environments")

    grid = env.all_states

    # Get the forward policy distribution for all states
    with torch.no_grad():
        # Handle both FM and other GFlowNet types
        policy: GFNModule = cast(
            GFNModule, gflownet.logF if isinstance(gflownet, FMGFlowNet) else gflownet.pf
        )

        estimator_outputs = policy(grid)
        dist = policy.to_probability_distribution(grid, estimator_outputs)
        probabilities = torch.exp(dist.logits)  # Get raw probabilities

    u = torch.ones(grid.batch_shape)

    indices = env.all_indices()
    for index in indices[1:]:
        parents = [
            tuple(list(index[:i]) + [index[i] - 1] + list(index[i + 1 :]) + [i])
            for i in range(len(index))
            if index[i] > 0
        ]
        parents_tensor = torch.tensor(parents)
        parents_indices = parents_tensor[:, :-1].long()  # All but last column for u
        action_indices = parents_tensor[:, -1].long()  # Last column for probabilities

        # Compute u values for parent states.
        parent_u_values = []
        for p in parents_indices:
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_u_values.append(u[grid_idx])
            # parent_u_values.append(u[tuple(p.tolist())])
            # # torch.all(grid.tensor == p, 1)
        parent_u_values = torch.stack(parent_u_values)
        # parent_u_values = torch.stack([u[tuple(p.tolist())] for p in parents_indices])

        # Compute probabilities for parent transitions.
        parent_probs = []
        for p, a in zip(parents_indices, action_indices):
            grid_idx = torch.all(grid.tensor == p, 1)  # index along flattened grid.
            parent_probs.append(probabilities[grid_idx, a])
        parent_probs = torch.stack(parent_probs)

        u[indices.index(index)] = torch.sum(parent_u_values * parent_probs)

    return (u * probabilities[..., -1]).detach().cpu()


def validate_hypergrid(
    env,
    gflownet,
    n_validation_samples,
    visited_terminating_states: DiscreteStates | None,
    discovered_modes,
):
    # Standard validation shared across envs.
    validation_info, visited_terminating_states = validate(
        env,
        gflownet,
        n_validation_samples,
        visited_terminating_states,
    )

    # Modes will have a reward greater than 1.
    mode_reward_threshold = 1.0  # Assumes height >= 5. TODO - verify.

    assert isinstance(visited_terminating_states, DiscreteStates)
    modes = visited_terminating_states[
        env.reward(visited_terminating_states) >= mode_reward_threshold
    ].tensor

    # Finds all the unique modes in visited_terminating_states.
    modes_found = set([tuple(s.tolist()) for s in modes])
    discovered_modes.update(modes_found)
    validation_info["n_modes_found"] = len(discovered_modes)
    return validation_info, visited_terminating_states, discovered_modes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument(
        "--ndim", type=int, default=4, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=16, help="Height of the environment"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_logz",
        type=float,
        default=1e-1,
        help="Learning rate for the logZ parameter",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--validation_interval", type=int, default=100, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=100000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    parser.add_argument(
        "--epsilon", type=float, default=0, help="Epsilon for the sampler"
    )
    parser.add_argument("--sampler", type=str, default="normal", help="Sampler type (normal, small_then_large, large_then_small)")
    
    args = parser.parse_args()

    main(args)
