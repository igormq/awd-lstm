import argparse
from clearml.automation import UniformParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from clearml import Task
import optuna


def job_complete_callback(
        job_id,  # type: str
        objective_value,  # type: float
        objective_iteration,  # type: int
        job_parameters,  # type: dict
        top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration,
          job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(
            objective_value))


hyper_parameters = {
    'awd': [
        DiscreteParameterRange('Args/gradient_clip_val',
                               values=(0, 0.25, 1., 400.)),
        UniformParameterRange('Args/output_dropout',
                              min_value=0.3,
                              max_value=0.6),
        UniformParameterRange('Args/hidden_dropout',
                              min_value=0.4,
                              max_value=0.6),
        UniformParameterRange('Args/input_dropout',
                              min_value=0.5,
                              max_value=0.7),
        UniformParameterRange('Args/embedding_dropout',
                              min_value=0.0,
                              max_value=0.5),
        UniformParameterRange('Args/weight_dropout',
                              min_value=0.5,
                              max_value=0.8),
        UniformParameterRange('Args/learning_rate',
                              min_value=0.01,
                              max_value=20),
        UniformParameterRange('Args/weight_decay', min_value=1e-9, max_value=1)
    ]
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='awd',
                        choices=['rnn', 'awd', 'transformer'])
    parser.add_argument('--template-task-id',
                        '--id',
                        '--task-id',
                        type=str,
                        required=True)
    parser.add_argument('--execution-queue',
                        '-q',
                        type=str,
                        default='rtx2080ti')
    parser.add_argument('--run-as-service', '--service', action="store_true")
    parser.add_argument('--no-reuse-last-task-id',
                        dest='reuse_id',
                        action="store_false",
                        default=True)
    args = parser.parse_args()

    task = Task.init(project_name='language-model-hp',
                     task_name=f'{args.model}',
                     task_type=Task.TaskTypes.optimizer,
                     reuse_last_task_id=args.reuse_id)
    task.connect(args)

    optimizer = HyperParameterOptimizer(
        base_task_id=args.
        template_task_id,  # This is the experiment we want to optimize
        # here we define the hyper-parameters to optimize
        hyper_parameters=hyper_parameters[args.model],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='val_ppl',
        objective_metric_series='val_ppl',
        objective_metric_sign='min',  # maximize or minimize the objective metric
        # setting optimizer - clearml supports GridSearch, RandomSearch, OptimizerBOHB and OptimizerOptuna
        optimizer_class=OptimizerOptuna,
        # Configuring optimization parameters
        execution_queue=args.
        execution_queue,  # queue to schedule the experiments for execution
        max_number_of_concurrent_tasks=2,  # number of concurrent experiments
        optimization_time_limit=
        None,  # set the time limit for the optimization process
        compute_time_limit=
        None,  # set the compute time limit (sum of execution time on all machines)
        total_max_jobs=
        1000,  # set the maximum number of experiments for the optimization. 
        # Converted to total number of iteration for OptimizerBOHB
        min_iteration_per_job=
        150000,  # minimum number of iterations per experiment, till early stopping
        max_iteration_per_job=
        200000,  # maximum number of iterations per experiment
        # optuna_sampler=optuna.samplers.TPESampler,
        # optuna_pruner=optuna.pruners.Hyperband)
    )

    if args.run_as_service:
        task.execute_remotely(queue_name='services', exit_process=True)

    # report every 12 seconds, this is way too often, but we are testing here
    optimizer.set_report_period(10)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    # optimizer.set_time_limit(in_minutes=120.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()
    print('We are done, good bye')