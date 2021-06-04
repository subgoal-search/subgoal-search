# from supervised.int.gen_subgoal_data import generate_problems
from supervised.rubik.rubik_solver_utils import generate_problems_rubik


class GoalBuilderRubik:
    def __init__(self,
                 generator_class=None,
                 policy_class=None):
        self.generator = generator_class()
        self.policy = policy_class()

    def construct_networks(self):
        self.generator.construct_networks()
        self.policy.construct_networks()

    def build_goals(self, current_state):
        raw_subgoals = self.generator.generate_subgoals(current_state)
        # raw_subgoals.add('$@yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww$')
        verifed_subgoals = []
        for raw_subgoal in raw_subgoals:
            # raw subgoal starts with '$@', should be changed to '?'
            raw_subgoal = '?' + raw_subgoal[2:]
            reached, path, done, current_proof_state = self.policy.reach_subgoal(current_state, raw_subgoal)
            if reached:
                verifed_subgoals.append((current_proof_state, path, done))
            if done:
                return verifed_subgoals, (current_proof_state, path, done)
        return verifed_subgoals, None


def goal_builder_test():
    PROOFS = 3
    tmp = GoalBuilderRubik()
    tmp.construct_networks()
    example_problems = generate_problems_rubik(PROOFS)

    for i in range(PROOFS):
        print(
            '______________________________________START_______________________________________________________________')
        example_problem = example_problems[i]
        subgoals, done = tmp.build_goals(example_problem[0])
        print(f'Subgoals {len(subgoals)}: {subgoals}')
    print(
        '_________________________________________END____________________________________________________________ \n \n')

# goal_builder_test()
