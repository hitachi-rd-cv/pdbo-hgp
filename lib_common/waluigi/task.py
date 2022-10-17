# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
from __future__ import division

import datetime
import os
from typing import Mapping

import gokart
import luigi
import torch

from lib_common.mysacred import Experiment
from lib_common.waluigi.tools import PyTorchPickleFileProcessor


class TaskBase(gokart.TaskOnKart):
    """TaskBase
    Base class inherited by most of Tasks

    Attributes:
        workspace_directory: root directory of the output it is insignificant for determining the name of output directly
    """
    workspace_directory: str = luigi.Parameter('./processed', significant=False)
    db_name: str = luigi.Parameter('no_name', significant=False)
    mongo_auth: str = luigi.Parameter(None, significant=False)
    memo: str = luigi.Parameter('none', significant=False)
    fix_random_seed_value = luigi.IntParameter(0)
    fix_random_seed_methods = luigi.ListParameter([
        "random.seed",
        "numpy.random.seed",
        "torch.random.manual_seed",
        "torch.cuda.manual_seed_all",
    ])
    name: str = luigi.Parameter('no_name', significant=False)
    _func_run: staticmethod

    @luigi.Task.event_handler(luigi.Event.START)
    def make_torch_deterministic(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run_in_sacred_experiment(self, f, **kwargs):
        ex = Experiment(self.__class__.__name__, db_name=self.db_name, mongo_auth=self.mongo_auth, base_dir=os.path.abspath(os.path.curdir))
        ex.main(f)
        names_significant_param = self.get_param_names(include_significant=True)
        param_kwargs_sig = {k: v for k, v in self.param_kwargs.items() if k in names_significant_param}
        ex.add_config(self.recursively_make_dict(param_kwargs_sig))
        ex.add_config({'seed': self.fix_random_seed_value})
        run = ex._create_run(bypassed_config=kwargs)
        return run()

    def output(self) -> object:
        '''
        do not overwrite this class in the child classes.
        this is executed in self.run to get unique output directly.
        Each combination of the task parameter leads its unique hash contained in self.task id.
        It enables output directory to be determined automatically and ensures the same task with different parameters are never overwritten.

        Returns: gokart.target.Target

        '''
        return self.make_target(f'{self.task_family}.pt', processor=PyTorchPickleFileProcessor())

    def make_and_get_temporary_directory(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        dir_tmp = os.path.join(self.workspace_directory, f'{self.task_family}_{self.task_unique_id}_{timestamp}')
        os.makedirs(dir_tmp, exist_ok=True)
        return dir_tmp

    @classmethod
    def recursively_make_dict(cls, value):
        if isinstance(value, Mapping):
            return dict(((k, cls.recursively_make_dict(v)) for k, v in value.items()))
        return value
