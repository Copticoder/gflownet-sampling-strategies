import math
from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet import DBGFlowNet
from gfn.modules import ConditionalScalarEstimator, GFNModule, ScalarEstimator
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
    warn_about_recalculating_logprobs,
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
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = True
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

        states = transitions.states
        actions = transitions.actions

        if len(states) == 0:
            return (
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(0.0, device=transitions.device),
            )

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
        # assert transitions.next_states.is_sink_state.equal(transitions.is_terminating)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_terminating]
        valid_transitions_is_terminating = transitions.is_terminating[
            ~transitions.states.is_sink_state
        ]

        if len(valid_next_states) == 0:
            return (
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(self.log_prob_min, device=transitions.device),
                torch.tensor(0.0, device=transitions.device),
            )

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(
                    valid_next_states,
                    transitions.conditioning[~transitions.is_terminating],
                ).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)

        log_F_s_next = torch.zeros_like(log_pb_actions)
        log_F_s_next[~valid_transitions_is_terminating] = valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        log_F_s_next[valid_transitions_is_terminating] = valid_transitions_log_rewards[
            valid_transitions_is_terminating
        ]
        targets = log_pb_actions + log_F_s_next

        scores = preds - targets

        assert scores.shape == (transitions.n_transitions,)
        return (log_pf_actions, log_pb_actions, scores)

    def loss(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = True
    ) -> torch.Tensor:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266).
        """
        warn_about_recalculating_logprobs(transitions, recalculate_all_logprobs)
        _, _, scores = self.get_scores(env, transitions, recalculate_all_logprobs)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
    def to_training_samples(self, trajectories: Trajectories) -> Transitions:        
        if trajectories.conditioning is not None:
            expand_dims = (trajectories.max_length,) + tuple(trajectories.conditioning.shape)
            conditioning = trajectories.conditioning.unsqueeze(0).expand(expand_dims)[
                ~trajectories.actions.is_dummy
            ]
        else:
            conditioning = None
        # get only last transitions from trajectories
        # get last states, and last_last states
        last_states = trajectories.states[trajectories.terminating_idx - 1, torch.arange(trajectories.n_trajectories)]
        last_last_states = trajectories.states[trajectories.terminating_idx - 2, torch.arange(trajectories.n_trajectories)]
        # get last states actions and last last states actions
        last_actions = trajectories.actions[-1]
        last_last_actions = trajectories.actions[-2]
        is_terminating = (
            last_states.is_sink_state
            if not trajectories.is_backward
            else last_states.is_initial_state
        )
        if trajectories._log_rewards is None:
            log_rewards = None
        else:
            log_rewards = torch.full(
                last_last_actions.batch_shape,
                fill_value=-float("inf"),
                dtype=torch.float,
                device=last_last_actions.device,
            )
            # TODO: Can we vectorize this?
            log_rewards[is_terminating] = torch.cat(
                [
                    trajectories._log_rewards[trajectories.terminating_idx == i]
                    for i in range(trajectories.terminating_idx.max() + 1)
                ],
                dim=0,
            )

        # Initialize log_probs None if not available
        if trajectories.has_log_probs:
            log_probs = trajectories.log_probs[~trajectories.actions.is_dummy]  # type: ignore
        else:
            log_probs = None

        return Transitions(
            env=trajectories.env,
            states=last_last_states,
            conditioning=conditioning,
            actions=last_last_actions,
            is_terminating=is_terminating,
            next_states=last_states,
            is_backward=trajectories.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
        )