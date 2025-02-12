from itertools import combinations_with_replacement, permutations
from gfn.gym import HyperGrid
from gfn.modules import GFNModule
import torch
import numpy as np
from matplotlib import pyplot as plt
def tupleOfSum(n: int, target: int, bound: int):
    for combination in combinations_with_replacement(range(min(bound + 1, target + 1)), n):
        if sum(combination) == target:
            for perm in set(permutations(combination)):
                yield perm
                
def terminatingProbsFromGfn(pf: GFNModule, env: HyperGrid, verbose:bool=False):
    arrival_probs = {tuple(state.tolist()): 0. for state in env.all_states.tensor}
    terminating_probs = {tuple(state.tolist()): 0. for state in env.all_states.tensor}
    arrival_probs[tuple(env.s0.tolist())] = 1.
    for i in range (env.ndim * (env.height-1)):
        for comb in tupleOfSum(env.ndim, i, env.height-1):
            state = env.states_from_tensor(torch.tensor(list(comb)).unsqueeze(0))
            masks = state.forward_masks
            logits = pf(state)
            logits[~masks] = -float("inf")
            output_probs = torch.softmax(logits, dim=-1)
            if verbose:
                print(output_probs)
            terminating_probs[tuple(comb)] = output_probs[0][-1].item() * arrival_probs[comb]
            for j in range(len(output_probs[0])-1):
                if comb[j]<env.height-1:
                    arrival_probs[tuple(comb[i]+1 if i == j else comb[i] for i in range(env.ndim))] += output_probs[0][j].item() * arrival_probs[comb]
    terminating_probs[env.ndim*(env.height-1,)] = arrival_probs[env.ndim*(env.height-1,)]
    return terminating_probs

def plotGrid(mainGFN: GFNModule, env: HyperGrid, plot: bool = True):
    coordinates = env.all_states.tensor
    real_rewards_list = env.reward(env.all_states)
    real_dist_list = env.true_dist_pmf
    real_dist = {tuple(state.tolist()): real_dist_list[i] for i, state in enumerate(env.all_states.tensor)}
    real_reward = {tuple(state.tolist()): real_rewards_list[i] for i, state in enumerate(env.all_states.tensor)}

    estimated_dist = terminatingProbsFromGfn(mainGFN.logF, env)
    estimated_dist_list = list(estimated_dist.values())
    if plot:
        grid_size = torch.max(coordinates, dim=0).values + 1
        grid_1 = np.zeros(grid_size.tolist())
        grid_2 = np.zeros(grid_size.tolist())

        for coord, intensity1, intensity2 in zip(coordinates, real_dist_list, estimated_dist_list):
            x, y = coord.tolist()
            grid_1[x, y] = intensity1
            grid_2[x, y] = intensity2

        # Plot side-by-side heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # First heatmap: Real distribution
        axes[0].imshow(grid_1, cmap="viridis", origin="upper")
        axes[0].set_title("Real distribution")
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")
        axes[0].set_ylim(axes[0].get_ylim()[::-1])

        # Second heatmap: Estimated distribution
        axes[1].imshow(grid_2, cmap="viridis", origin="upper")
        axes[1].set_title("Estimated distribution")
        axes[1].set_xlabel("X Coordinate")
        axes[1].set_ylabel("Y Coordinate")
        axes[1].set_ylim(axes[1].get_ylim()[::-1])

        # Add colorbars
        fig.colorbar(axes[0].imshow(grid_1, cmap='viridis', origin='upper'), ax=axes[0], orientation='vertical')
        fig.colorbar(axes[1].imshow(grid_2, cmap='viridis', origin='upper'), ax=axes[1], orientation='vertical')

        plt.tight_layout()
        plt.show()
