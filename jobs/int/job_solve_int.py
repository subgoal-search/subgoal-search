import pickle
import time
from copy import deepcopy

from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator, log_text
from supervised.int.gen_subgoal_data import generate_problems

from visualization.seq_parse import logic_statement_to_seq_string, entity_to_seq_string


def solve_problem(solver, input_state):

    time_s = time.time()
    solution, tree_metrics, root, trajectory_actions, additional_info = solver.solve(input_state)
    time_solving = time.time() - time_s
    return dict(
        solution=solution,
        tree_metrics=tree_metrics,
        root=root,
        trajectory_actions=trajectory_actions,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        additional_info=additional_info
    )


class JobSolveINT(Job):
    def __init__(self,
                 solver_class,
                 n_jobs,
                 n_parallel_workers,
                 batch_size,
                 budget_checkpoints=None,
                 log_solutions_limit=100,
                 job_range = None,
                 collect_solutions=None
                 ):

        self.solver_class = solver_class
        self.n_jobs = n_jobs
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.budget_checkpoints = budget_checkpoints
        self.log_solutions_limit = log_solutions_limit
        self.job_range = job_range
        self.collect_solution = collect_solutions

        self.solved_stats = MetricsAccumulator()
        self.experiment_stats = MetricsAccumulator()

        self.logged_solutions = 0

        if self.collect_solution is not None:
            self.collection = {}


    def execute(self):

        proofs_to_solve = generate_problems(self.n_jobs)

        solver = self.solver_class()
        solver.construct_networks()
        jobs_done = 0


        total_time_start = time.time()
        for job_num in range(self.n_jobs):
            print('============================ Solving {:>4}  out  of  {:>4} ============================'.
                  format(job_num, self.n_jobs))
            results = [solve_problem(solver, proofs_to_solve[job_num][0])]

            print('===================================================================================')
            self.log_results(results, jobs_done)
            jobs_done += 1

        for metric, value in self.solved_stats.return_scalars().items():
            log_text('summary', f'{metric},  {value}')
        log_text('summary', f'Finished time , {time.time() - total_time_start}')

    def log_results(self, results, step):

        n_logs = len(results)
        for log_num, result in enumerate(results):
            log_scalar_metrics('tree', step+log_num, result['tree_metrics'])
            if self.logged_solutions < self.log_solutions_limit:
                self.log_solution(result['solution'], result['trajectory_actions'], result['input_problem'], step+log_num)
            solved = result['solution'] is not None
            self.experiment_stats.log_metric_to_accumulate('tested', 1)
            log_scalar_metrics('problems', step+log_num, self.experiment_stats.return_scalars())
            if solved:
                self.solved_stats.log_metric_to_average('rate', 1)
                self.solved_stats.log_metric_to_accumulate('problems', 1)
                log_scalar('solution', step + log_num, 1)
                log_scalar('solution/length', step + log_num, len(result['trajectory_actions']))
                # assert False
                trajectory_actions = [str(action) for action in result['trajectory_actions']]
                trajectory = ', '.join(trajectory_actions)
                log_text('trajectory_actions', f'{step + log_num}: {trajectory}', False)
                log_scalar('solution/n_subgoals', step + log_num, len(result['solution']))
            else:
                self.solved_stats.log_metric_to_average('rate', 0)
                self.solved_stats.log_metric_to_accumulate('problems', 0)
                log_scalar('solution', step+log_num, 0)
                log_scalar('solution/length', step + log_num, -1)
                log_text('trajectory_actions', f'{step + log_num}: unsolved', False)
                log_scalar('solution/n_subgoals', step + log_num, -1)

            log_scalar_metrics('predictions', step+log_num, result['additional_info']['predictions'])
            # log_scalar('problems', step + n_logs, step + n_logs)


            if self.budget_checkpoints is not None:
                for budget in self.budget_checkpoints:
                    if result['tree_metrics']['expanded_nodes'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_exp_nodes', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_exp_nodes', 0)

                    if result['tree_metrics']['nodes'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', 0)


        log_scalar_metrics('solved', step+n_logs, self.solved_stats.return_scalars())


    def log_solution(self, solution, trajectory_actions, input_problem, step):

        if solution is not None:
            solution_str = f'Problem {step} : {solution[0].hash} \n'
            for subgoal_num, node in enumerate(solution[1:]):
                solution_str += f'subgoal {subgoal_num} : {node.hash} \n'
            solution_str += '\n \n'
            solution_str += 'Actions: \n'
            for action_num, action in enumerate(trajectory_actions):
                solution_str += f'action {action_num}: ({action[0]}, {[entity_to_seq_string(ent) for ent in action[1:]]} ) \n'

        else:
            solution_str = f'Unsolved problem {step} : {logic_statement_to_seq_string(input_problem["observation"]["objectives"][0])} \n \n'

        log_text('solution', solution_str, True)

