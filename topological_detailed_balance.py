import math
from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet import DBGFlowNet
from gfn.modules import ConditionalScalarEstimator, GFNModule, ScalarEstimator
from gfn.utils.common import has_log_probs
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)
from gfn.utils.prob_calculations import get_transition_pfs_and_pbs
from gfn.containers import Transitions

def check_compatibility(states, actions, transitions):
    if states.batch_shape != tuple(actions.batch_shape):
        if type(transitions) is not Transitions:
            raise TypeError(
                "`transitions` is type={}, not Transitions".format(type(transitions))
            )
        else:
            raise ValueError(" wrong happening with log_pf evaluations")


class TopologicalDBGFlowNet(DBGFlowNet):
    r"""The Detailed Balance GFlowNet.

    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times
    \mathcal{O}_3$, where $\mathcal{O}_1$ is the set of functions from the internal
    states (no $s_f$) to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the
    non-negativity constraint), and $\mathcal{O}_2$ is the set of forward probability
    functions consistent with the DAG. $\mathcal{O}_3$ is the set of backward
    probability functions consistent with the DAG, or a singleton thereof, if
    `self.logit_PB` is a fixed `DiscretePBEstimator`.

    Attributes:
        logF: a ScalarEstimator instance.
        forward_looking: whether to implement the forward looking GFN loss.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """


    def get_scores(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Returns: A tuple of three tensors of shapes (n_transitions,), representing the
            log probabilities of the actions, the log probabilities of the backward actions, and th scores.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        # only get last transitions 
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)
        check_compatibility(states, actions, transitions)

        log_pf_actions, log_pb_actions = self.get_pfs_and_pbs(
            transitions, recalculate_all_logprobs
        )

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                log_F_s = self.logF(states, transitions.conditioning).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                log_F_s = self.logF(states).squeeze(-1)

        if self.forward_looking:
            log_rewards = env.log_reward(states)  # TODO: RM unsqueeze(-1) ?
            if math.isfinite(self.log_reward_clip_min):
                log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
            log_F_s = log_F_s + log_rewards

        preds = log_pf_actions + log_F_s

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(
                    valid_next_states, transitions.conditioning[~transitions.is_done]
                ).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)

        log_F_s_next = torch.zeros_like(log_pb_actions)
        log_F_s_next[~valid_transitions_is_done] = valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        log_F_s_next[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]
        targets = log_pb_actions + log_F_s_next

        scores = preds - targets

        assert scores.shape == (transitions.n_transitions,)
        return log_pf_actions, log_pb_actions, scores

    def loss(self, env: Env, transitions: Transitions) -> torch.Tensor:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)."""
        _, _, scores = self.get_scores(env, transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        # get only last transitions from trajectories
        # get last states, and last_last states
        last_states = trajectories.states[trajectories.when_is_done - 1, torch.arange(trajectories.n_trajectories)]
        last_last_states = trajectories.states[trajectories.when_is_done - 2, torch.arange(trajectories.n_trajectories)]
        # get last states actions and last last states actions
        last_actions = trajectories.actions[-1]
        
        last_last_actions = trajectories.actions[-2]
        
        is_done = (
            last_states.is_sink_state
            if not self.is_backward
            else last_states.is_initial_state
        )
        if self._log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full(
                actions.batch_shape,
                fill_value=-float("inf"),
                dtype=torch.float,
                device=actions.device,
            )
            log_rewards[is_done] = torch.cat(
                [
                    self._log_rewards[self.when_is_done == i]
                    for i in range(self.when_is_done.max() + 1)
                ],
                dim=0,
            )

        # Only return logprobs if they exist.
        log_probs = (
            self.log_probs[~self.actions.is_dummy] if has_log_probs(self) else None
        )

        return Transitions(
            env=self.env,
            states=last_last_states,
            conditioning=conditioning,
            actions=actions,
            is_done=is_done,
            next_states=last_states,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
        )