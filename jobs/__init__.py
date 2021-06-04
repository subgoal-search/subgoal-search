import gin

from jobs import (
    job_create_samples_sokoban,
    job_draw_goal_building_sokoban,
    job_solve_sokoban,
    job_solve_sokoban_pixel_diff,
    job_train_sokoban,
    job_train_sokoban_pixel_diff,
)

from jobs.int import job_int_generate_goal_hf, job_solve_int, job_vanilla_solve_int
from jobs.int import job_int_generate_problems
from jobs.int import job_int_train_goal_hf
from jobs.int import job_int_train_policy_hf_pointer
from jobs.int import job_int_train_goal_lstm
from jobs.int import job_int_sample_from_lstm
from jobs.int import job_int_train_policy_lstm
from jobs.int import job_int_train_value
from jobs.int import job_mcts_solve_int
from jobs.int import job_vanilla
from jobs.int import job_shooting_solve_int

from jobs.rubik import job_rubik_train_goal_hf
from jobs.rubik import job_rubik_train_value_hf
from jobs.rubik import job_rubik_train_policy_hf
from jobs.rubik import job_rubik_validate_policy_hf
from jobs.rubik import job_solve_rubik
from jobs.rubik import job_rubik_shooting_hf


def configure_job(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='jobs'
    )


JobSolveSokoban = configure_job(job_solve_sokoban.JobSolveSokoban)
JobSolveSokobanPixelDiff = configure_job(job_solve_sokoban_pixel_diff.JobSolveSokobanPixelDiff)
JobTrainSokoban = configure_job(job_train_sokoban.JobTrainSokoban)
JobTrainSokobanPixelDiff = configure_job(job_train_sokoban_pixel_diff.JobTrainSokobanPixelDiff)
JobCreateSamplesSokoban = configure_job(job_create_samples_sokoban.JobCreateSamplesSokoban)
JobDrawGoalBuildingSokoban = configure_job(job_draw_goal_building_sokoban.JobDrawGoalBuildingSokoban)

JobIntTrainGoalLSTM = configure_job(job_int_train_goal_lstm.JobIntTrainGoalLSTM)
JobIntSampleFromLSTM = configure_job(job_int_sample_from_lstm.JobIntSampleFromLSTM)
JobIntTrainPolicyLSTM = configure_job(job_int_train_policy_lstm.JobIntTrainPolicyLSTM)

TrainHfForRubikGoal = configure_job(job_rubik_train_goal_hf.TrainHfForRubikGoal)
TrainHfForRubikValue = configure_job(job_rubik_train_value_hf.TrainHfForRubikValue)
TrainHfForRubikPolicy = configure_job(job_rubik_train_policy_hf.TrainHfForRubikPolicy)
TrainHfForRubikValidatePolicy = configure_job(job_rubik_validate_policy_hf.TrainHfForRubikValidatePolicy)
JobSolveRubik = configure_job(job_solve_rubik.JobSolveRubik)
JobShootingSolveRubik = configure_job(job_rubik_shooting_hf.JobShootingSolveRubik)

JobIntGenerateGoalHf = configure_job(job_int_generate_goal_hf.JobIntGenerateGoalHf)
TrainHfForIntGoal = configure_job(job_int_train_goal_hf.TrainHfForIntGoal)
JobIntGenerateProblems = configure_job(job_int_generate_problems.JobIntGenerateProblems)
JobMCTSSolveInt = configure_job(job_mcts_solve_int.JobMCTSSolveInt)

TrainHfForIntPolicy = configure_job(job_int_train_policy_hf_pointer.TrainHfForIntPolicyPointer)
TrainHfForIntValue = configure_job(job_int_train_value.TrainHfForIntValue)
JobSolveINT = configure_job(job_solve_int.JobSolveINT)

JobVanilla = configure_job(job_vanilla.JobVanilla)
JobVanillaSolveINT = configure_job(job_vanilla_solve_int.JobVanillaSolveINT)

JobShootingSolveINT = configure_job(job_shooting_solve_int.JobShootingSolveINT)