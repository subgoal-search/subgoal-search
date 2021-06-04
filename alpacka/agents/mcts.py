
import math

import gin
import numpy as np

from alpacka.agents import tree_search


@gin.configurable
def puct_exploration_bonus(child_count, parent_count, prior_probability):
    return math.sqrt(parent_count) / (child_count + 1) * prior_probability


class Node(tree_search.Node):

    def __init__(self, prior_probability, state):
        super().__init__(state=state)
        self.prior_probability = prior_probability


class MCTSAgent(tree_search.TreeSearchAgent):

    def __init__(
        self,
        exploration_bonus_fn=puct_exploration_bonus,
        exploration_weight=1.0,
        sampling_temperature=0.0,
        prior_noise_weight=0.0,
        prior_noise_parameter=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._exploration_bonus = exploration_bonus_fn
        self._exploration_weight = exploration_weight
        self._sampling_temperature = sampling_temperature
        self._prior_noise_weight = prior_noise_weight
        self._prior_noise_parameter = prior_noise_parameter

    def _choose_action(self, node, actions, exploratory):
        def rate_child(child):
            if exploratory:
                quality = child.quality(self._discount) + (
                    self._exploration_weight * self._exploration_bonus(
                        child.count, node.count, child.prior_probability
                    )
                )
            else:
                quality = np.log(child.count) if child.count else float('-inf')


                u = np.random.uniform(low=1e-6, high=1.0 - 1e-6)
                g = -np.log(-np.log(u))
                quality += g * self._sampling_temperature
            return quality

        child_qualities_and_actions = [
            (rate_child(node.children[action]), action) for action in actions
        ]
        (_, action) = max(child_qualities_and_actions)
        return action

    def _on_new_root(self, root):
        prior = np.array([child.prior_probability for child in root.children])
        noise = np.random.dirichlet(
            [self._prior_noise_parameter] * len(root.children)
        )
        prior = (
            (1 - self._prior_noise_weight) * prior +
            self._prior_noise_weight * noise
        )
        for (child, p) in zip(root.children, prior):
            child.prior_probability = p

    def _compute_node_info(self, node):
        node_info = super()._compute_node_info(node)
        qualities = node_info['qualities']
        prior_probabilities = np.array([
            child.prior_probability for child in node.children
        ])
        exploration_bonuses = self._exploration_weight * np.array([
            self._exploration_bonus(
                child.count, node.count, child.prior_probability
            )
            for child in node.children
        ])
        total_scores = qualities + exploration_bonuses
        return {
            **node_info,
            'prior_probabilities': prior_probabilities,
            'exploration_bonuses': exploration_bonuses,
            'total_scores': total_scores,
        }


def uniform_prior(n):
    return np.full(shape=(n,), fill_value=(1 / n))

