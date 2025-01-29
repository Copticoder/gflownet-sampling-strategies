# import torch
from typing import Any, Tuple
import wandb
from tqdm import tqdm, trange

from gfn.gflownet import (
    FMGFlowNet,
)
from GrowingTriangleSampler import GrowingTriangleSampler
from gfn.gym import HyperGrid
from gfn.actions import Actions
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP
from gfn.utils.training import validate
import torch

from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs

DEFAULT_SEED = 4444
# Set condition to only allow exit action for states at height-1
def main(args):  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # 1. Create the environment
    
    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        R0=args.R0,
        R1=args.R1,
        R2=args.R2,
        device_str=device_str,
    )
    if not args.uniform_pb:
        pb_module = MLP(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=pf_module.trunk if args.tied else None,
        )
    module = MLP(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
    estimator = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )
    gflownet = FMGFlowNet(estimator)
    
    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # 3. Create the optimizer
    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.states_from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    validation_info = {"l1_dist": float("inf")}
    # Training loop
    growing_triangle_sampler = GrowingTriangleSampler(estimator, n_iterations)
    for iteration in trange(1,n_iterations+1):
            
        # Sample trajectories
        trajectories = growing_triangle_sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
        )        
        training_samples = gflownet.to_training_samples(trajectories)
        
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)
        loss.backward()
        optimizer.step()
        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
    
        if iteration % args.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")

    return validation_info["l1_dist"]

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )

    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
        + " neural network modules",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    args = parser.parse_args()

    print(main(args))



