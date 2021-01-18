from pathlib import Path

from nni.experiment import Experiment, RemoteMachineConfig
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

tuner = HyperoptTuner('tpe', optimize_mode='maximize')

search_space = {
    "learning_rate": {
        "_type": "loguniform",
        "_value": [1e-2, 20]
    },
    "output_dropout": {
        "_type": "uniform",
        "_value": [0.3, 0.6]
    },
    "hidden_dropout": {
        "_type": "uniform",
        "_value": [0.4, 0.6]
    },
    "input_dropout": {
        "_type": "uniform",
        "_value": [0.5, 0.7]
    },
    "embedding_dropout": {
        "_type": "uniform",
        "_value": [0, 0.5]
    },
    "weight_dropout": {
        "_type": "uniform",
        "_value": [0.5, 0.8]
    },
    "alpha": {
        "_type": "choice",
        "_value": [0, 2]
    },
    "beta": {
        "_type": "choice",
        "_value": [0, 1]
    },
    "weight_decay": {
        "_type": "loguniform",
        "_value": [1e-9, 1]
    },
    "gradient_clip_val": {
        "_type": "choice",
        "_value": [0, 0.25]
    }
}

experiment = Experiment(tuner, ['local', 'remote'])
experiment.config.experiment_name = 'awd_lstm'
experiment.config.author_name = 'Igor Quintanilha'
experiment.config.max_trial_number = 100
experiment.config.max_experiment_duration = '60d'

experiment.config.nni_manager_ip = '10.221.90.21'
experiment.config.search_space = search_space

experiment.config.trial_prepare_command = 'source /home/igor.quintanilha/miniconda3/bin/activate dsc'
experiment.config.trial_command = 'python main.py --gpus 1 data/brtd --vocab data/brtd/b3922f0904f4f1b7b258a9488132f2e6480cf936493be53f74fd7aaa07e14781.8f9337.vocab --batch-size 64 --max_epochs 10 --terminate_on_nan --num-embedding 400 --num-layers 3 --num-hidden 1150 --model awd --bptt 20 --max_steps 150000 --val_check_interval .25'
experiment.config.trial_code_directory = Path(__file__).parent.parent
experiment.config.trial_concurrency = 2
experiment.config.trial_gpu_number = 1

experiment.config.training_service[0].use_active_gpu = True
experiment.config.training_service[0].max_trial_number_per_gpu = True

experiment.config.training_service[1].reuse_mode = True

remote_confs = []
for ip in ['10.221.70.3', '10.221.70.15', '10.221.90.20']:
    rm_conf = RemoteMachineConfig()
    rm_conf.host = ip
    rm_conf.user = 'igor.quintanilha'
    rm_conf.ssh_key_file = '/home/igor.quintanilha/.ssh/id_rsa'
    rm_conf.use_active_gpu = True
    rm_conf.max_trial_number_per_gpu = 1
    remote_confs.append(rm_conf)

experiment.config.training_service[1].machine_list = remote_confs

experiment.start(26780, debug=False)