from supervised.int.gen_subgoal_data import generate_problems


class GoalBuilderINT:
    def __init__(self,
                 generator_class=None,
                 policy_class=None
                 ):

        self.generator = generator_class()
        self.policy = policy_class()

    def reset_counter(self):
        self.generator.reset_counter()
        self.policy.reset_counter()

    def read_counter(self):
        return {
            'generator': self.generator.read_counter(),
            'policy': self.policy.read_counter()
        }

    def construct_networks(self):
        self.generator.construct_networks()
        self.policy.construct_networks()


    def build_goals(self, current_state):
        raw_subgoals = self.generator.generate_subgoals(current_state)
        subgoal_strs = [raw_subgoal[0] for raw_subgoal in raw_subgoals]
        results = self.policy.reach_subgoals(current_state, subgoal_strs)
        return [subgoal for subgoal in results if subgoal is not None]

