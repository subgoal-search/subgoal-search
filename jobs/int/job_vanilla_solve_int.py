import pickle
import time
from copy import deepcopy

import cloudpickle

from envs import Sokoban

from joblib import Parallel, delayed
from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator, log_text
from supervised.int.gen_subgoal_data import generate_problems

from utils.general_utils import readable_num
from utils.utils_sokoban import draw_and_log
from visualization.seq_parse import logic_statement_to_seq_string, entity_to_seq_string


def solve_problem(vanilla_policy, input_state):
    time_s = time.time()
    solved, additional_info = vanilla_policy.solve(input_state)
    time_solving = time.time() - time_s
    return dict(
        solved=solved,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        additional_info=additional_info
    )


class JobVanillaSolveINT(Job):
    def __init__(self,
                 n_jobs,
                 vanilla_policy_class=None,
                 budget_checkpoints=None,
                 log_solutions_limit=100,
                 ):

        self.vanilla_policy = vanilla_policy_class()
        self.n_jobs = n_jobs
        self.budget_checkpoints = budget_checkpoints
        self.log_solutions_limit = log_solutions_limit

        self.solved_stats = MetricsAccumulator()
        self.experiment_stats = MetricsAccumulator()


        self.collection = {}


    def execute(self):

        self.vanilla_policy.construct_networks()
        proofs_to_solve = generate_problems(self.n_jobs)
        jobs_done = 0

        total_time_start = time.time()
        for job_num in range(self.n_jobs):
            print(f'============================ Problem {job_num} ============================')
            results = solve_problem(self.vanilla_policy, proofs_to_solve[job_num][0])
            print('===================================================================================')
            self.log_results(results, jobs_done)
            jobs_done += 1

        for metric, value in self.solved_stats.return_scalars().items():
            log_text('summary', f'{metric},  {value}')
        log_text('summary', f'Finished time , {time.time() - total_time_start}')

    def log_results(self, results, step):


        if results['solved']:
            self.solved_stats.log_metric_to_average('rate', 1)
            self.solved_stats.log_metric_to_accumulate('problems', 1)
            log_scalar('solution', step + 1, 1)

        else:
            self.solved_stats.log_metric_to_average('rate', 0)
            self.solved_stats.log_metric_to_accumulate('problems', 0)
            log_scalar('solution', step+1, 0)



        log_scalar_metrics('solved', step+1, self.solved_stats.return_scalars())




