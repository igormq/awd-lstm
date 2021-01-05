import argparse
from clearml.automation import UniformParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from clearml import Task

hyper_parameters = {
    'awd': [
        DiscreteParameterRange('gradient_clip_val',
                               values=(0, 0.25, 1., 400.)),
        UniformParameterRange('output_dropout', min_value=0.3, max_value=0.6),
        UniformParameterRange('hidden_dropout', min_value=0.4, max_value=0.6),
        UniformParameterRange('input_dropout', min_value=0.5, max_value=0.7),
        UniformParameterRange('embedding_dropout',
                              min_value=0.0,
                              max_value=0.5),
        UniformParameterRange('weight_dropout', min_value=0.5, max_value=0.8),
        UniformParameterRange('learning_rate', min_value=0.01, max_value=20),
        UniformParameterRange('weight_decay', min_value=1e-9, max_value=1)
    ]
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='awd',
                        choices=['rnn', 'awd', 'transformer'])
    parser.add_argument('--task-id', type=str, required=True)
    parser.add_argument('--execution-queue',
                        '-q',
                        type=str,
                        default='rtx2080ti')
    args = parser.parse_args()

    task = Task.init(project_name='language-model',
                     task_name=f'hp-{args.model}',
                     task_type=Task.TaskTypes.optimizer)

    optimizer = HyperParameterOptimizer(
        base_task_id=args.
        task_id,  # This is the experiment we want to optimize
        # here we define the hyper-parameters to optimize
        hyper_parameters=[hyper_parameters[args.model]],
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
        100,  # set the maximum number of experiments for the optimization. 
        # Converted to total number of iteration for OptimizerBOHB
        min_iteration_per_job=
        15000,  # minimum number of iterations per experiment, till early stopping
        max_iteration_per_job=
        1500000,  # maximum number of iterations per experiment
    )

    task.execute_remotely(queue_name=args.execution_queue, exit_process=True)
