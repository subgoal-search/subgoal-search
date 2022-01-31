from queue import PriorityQueue
import random

import numpy as np

from envs import Sokoban
from solvers.core import Solver
from utils.general_utils import readable_num
from utils.utils_sokoban import get_field_index_from_name, HashableNumpyArray


class SolverNode:
    def __init__(self, state, parent, p, depth, child_num, path):
        self.state = state
        self.parent = parent
        self.p = p
        self.depth = depth
        self.child_num = child_num
        self.path = path
        self.children = []

        # Attributes needed for drawing tree:
        self.expanded = False
        self.id = None
        self.level = 0

        if self.parent is not None:
            self.trajectory_p = self.parent.trajectory_p * self.p
        else:
            self.trajectory_p = self.p

    def add_child(self, child):
        self.children.append(child)

    def set_value(self, value):
        self.value = value


class GeneralSolverSokoban(Solver):
    def __init__(self):
        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()
        self.num_boxes = self.core_env.get_num_boxes()

    # Sokoban-specific:
    def solved(self, state):
        box_on_target = 0
        for x in range(self.dim_room[0]):
            for y in range(self.dim_room[1]):
                if np.argmax(state[x][y]) == get_field_index_from_name('box_on_goal'):
                    box_on_target += 1
        return box_on_target == self.num_boxes

class BestFSSolverSokoban(GeneralSolverSokoban):
    def __init__(self,
                 goal_builder_class,
                 value_estimator_class,
                 max_steps,
                 total_confidence_level,
                 internal_confidence_level,
                 max_goals,
                 max_tree_size,
                 max_tree_depth
                 ):
        super().__init__()
        self.max_steps = max_steps
        self.total_confidence_level = total_confidence_level
        self.internal_confidence_level = internal_confidence_level
        self.max_goals = max_goals
        self.max_tree_size = max_tree_size
        self.max_tree_depth = max_tree_depth

        self.goal_builder =  goal_builder_class()
        self.value_estimator = value_estimator_class()

    def construct_networks(self):
        self.value_estimator.construct_networks()
        self.goal_builder.construct_networks()

    def solve(self, input, collect_data_for_graph_tracer=False):

        assert self.value_estimator is not None, 'you must load value estimator'

        # Data needed for tree drawing (used only if collect_data_for_graph_tracer is True):
        tree_nodes = {}
        tree_edges = []
        tree_extra_edges = []
        nodes_constructed = 0

        solved = False
        root = SolverNode(input, None, 1, 0, 0, [])
        nodes_queue = PriorityQueue()
        # To prevent situations where two exactly same states (thus with the same value),
        # cannot be compared, we add another dimension
        # with random number which are being compared in these rare situations.
        root_value = self.value_estimator.evaluate(root.state)
        root.set_value(root_value)
        nodes_queue.put((-root_value, random.random(), root))
        solution = []
        tree_size = 1
        expanded_nodes = 0
        all_goals_created = 0
        tree_depth = 0
        total_path_between_goals = 0
        seen_hashed_states = {HashableNumpyArray(root.state)}
        hashed_state_to_id = {}

        if collect_data_for_graph_tracer:
            root.id = nodes_constructed
            nodes_constructed += 1
            root.level = 0
            tree_nodes[root.id] = root
            hashed_root_state = HashableNumpyArray(root.state)
            hashed_state_to_id[hashed_root_state] = root.id

        while True:
            if nodes_queue.empty():
                finished_cause = 'Finished cause queue is empty'
                break
            if tree_size >= self.max_tree_size:
                finished_cause = 'Finished cause tree too big'
                break
            if solved:
                finished_cause =  'Finished cause solved'
                break

            #pop node from queue to expand
            current_node = nodes_queue.get()[-1]
            reverse_order = True # We want goals returned by goal builder to be sorted from most to least probable.

            if current_node.depth < self.max_tree_depth:
                goals = self.goal_builder.build_goals(
                    current_node.state,
                    self.max_steps,
                    self.total_confidence_level,
                    self.internal_confidence_level,
                    self.max_goals,
                    reverse_order)

                if collect_data_for_graph_tracer:
                    current_node.expanded = True

                all_goals_created += len(goals)
                expanded_nodes += 1
                created_new = 0
                for child_num, goal_proposition in enumerate(goals):
                    current_goal_state = goal_proposition.goal_state
                    current_goal_state_hashed = goal_proposition.hashed_goal
                    path = goal_proposition.path
                    total_path_between_goals += len(path)

                    if current_goal_state_hashed not in seen_hashed_states:
                        created_new += 1
                        seen_hashed_states.add(current_goal_state_hashed)
                        new_node = SolverNode(current_goal_state, current_node, goal_proposition.p, current_node.depth + 1, child_num, path)
                        current_node.add_child(new_node)
                        tree_depth = max(tree_depth, new_node.depth)
                        node_val = self.value_estimator.evaluate(new_node.state)
                        new_node.set_value(node_val)
                        nodes_queue.put((-node_val, random.random(), new_node))
                        tree_size += 1

                        if collect_data_for_graph_tracer:
                            new_node.id = nodes_constructed
                            nodes_constructed += 1
                            new_node.level = new_node.parent.level + 1
                            tree_nodes[new_node.id] = new_node
                            tree_edges.append((current_node.id, new_node.id, readable_num(new_node.p)))
                            hashed_state_to_id[current_goal_state_hashed] = new_node.id

                        #look for solution
                        if self.solved(current_goal_state):
                            solution.append(new_node)
                            solved = True
                            break

                    else:
                        if collect_data_for_graph_tracer:
                            extra_edge_target = hashed_state_to_id[current_goal_state_hashed]
                            tree_extra_edges.append((current_node.id, extra_edge_target, readable_num(goal_proposition.p)))

        tree_metrics = {'nodes' : tree_size,
                        'expanded_nodes': expanded_nodes,
                        'unexpanded_nodes': tree_size - expanded_nodes,
                        'max_depth' : tree_depth,
                        'avg_n_goals': all_goals_created/expanded_nodes if expanded_nodes > 0 else 0,
                        'avg_dist_between_goals' : total_path_between_goals/all_goals_created
                        if all_goals_created > 0 else 0
                        }

        additional_info = dict(
            finished_cause=finished_cause,
            tree_nodes=tree_nodes,
            tree_edges=tree_edges,
            tree_extra_edges=tree_extra_edges
        )
        if solved:
            node = solution[0]
            while node.parent is not None:
                solution.append(node.parent)
                node = node.parent

            trajectory_actions = []
            for inter_goal in solution:
                trajectory_actions += list(inter_goal.path)

            inter_goals = [node for node in reversed(solution)]

            return (inter_goals, tree_metrics, root, trajectory_actions, additional_info)
        else:
            return (None, tree_metrics, root, None, additional_info)
