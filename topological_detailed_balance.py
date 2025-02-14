import math
from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.modules import ConditionalScalarEstimator, GFNModule, ScalarEstimator
from gfn.utils.common import has_log_probs
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)
from gfn.utils.prob_calculations import get_transition_pfs_and_pbs


def check_compatibility(states, actions, transitions):
    if states.batch_shape != tuple(actions.batch_shape):
        if type(transitions) is not Transitions:
            raise TypeError(
                "`transitions` is type={}, not Transitions".format(type(transitions))
            )
        else:
            raise ValueError(" wrong happening with log_pf evaluations")

class ModifiedDBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Modified Detailed Balance GFlowNet. Only applicable to environments where
    all states are terminating.

    See Bayesian Structure Learning with Generative Flow Networks
    https://arxiv.org/abs/2202.13903 for more details.
    """

    def get_scores(
        self, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> torch.Tensor:
        """DAG-GFN-style detailed balance, when all states are connected to the sink.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Raises:
            ValueError: when backward transitions are supplied (not supported).
            ValueError: when the computed scores contain `inf`.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]

        check_compatibility(states, actions, transitions)

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(states, transitions.conditioning[mask])
        else:
            with no_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(states)

        pf_dist = self.pf.to_probability_distribution(states, module_output)

        if has_log_probs(transitions) and not recalculate_all_logprobs:
            valid_log_pf_actions = transitions[mask].log_probs
        else:
            # Evaluate the log PF of the actions sampled off policy.
            valid_log_pf_actions = pf_dist.log_prob(actions.tensor)
        valid_log_pf_s_exit = pf_dist.log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0])
        )

        # The following two lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(valid_next_states)

        valid_log_pf_s_prime_exit = self.pf.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(torch.full_like(actions.tensor, actions.__class__.exit_action[0]))

        non_exit_actions = actions[~actions.is_exit]

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(valid_next_states)

        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

        scores = preds - targets
        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores

    def loss(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> torch.Tensor:
        """Calculates the modified detailed balance loss."""
        scores = self.get_scores(
            transitions, recalculate_all_logprobs=recalculate_all_logprobs
        )
        return torch.mean(scores**2)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
