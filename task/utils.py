import datetime
import itertools
import os
import shutil
from copy import deepcopy

import gokart
import luigi
from luigi import ListParameter
from sacred.utils import recursive_update

import lib_common.waluigi.tools
from lib_common import waluigi


class MoveOutputs(gokart.TaskOnKart):
    target_task = luigi.TaskParameter()
    removes = waluigi.MyListParameter(None)
    remove_all: bool = luigi.BoolParameter()
    workspace_directory = luigi.Parameter()
    timestamp = luigi.Parameter(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), significant=False)

    def requires(self):
        backup_dir = os.path.join(self.workspace_directory, 'removed_caches', self.timestamp)

        # recursively find the tasks between target_task and each task in self.removes
        if self.remove_all:
            d_task_removed = lib_common.waluigi.tools.get_downstream_tasks_recur(self.target_task, query_type='any')
            # add target_task itself to removing list
            d_task_removed.update({self.target_task.task_id: self.target_task})
        else:
            d_task_removed = {}
            for task in self.removes:
                d_task_removed_tmp = lib_common.waluigi.tools.get_downstream_tasks_recur(self.target_task, task)
                if len(d_task_removed_tmp) == 0:
                    raise ValueError(f'task "{task}" was not found in the task tree of "{self.target_task.task_family}"')
                d_task_removed.update(d_task_removed_tmp)

            # add target_task itself to removing list
            if self.target_task.task_family in self.removes:
                d_task_removed.update({self.target_task.task_id: self.target_task})

        requires = []
        for task in d_task_removed.values():
            if os.path.exists(task.output().path()):
                src = task.output().path()
                requires.append(Move(src=task.output().path(), dst=os.path.join(backup_dir, os.path.basename(src))))
                os.makedirs(backup_dir, exist_ok=True)

        return requires


class Move(gokart.TaskOnKart):
    dst = luigi.Parameter(significant=False)
    src = luigi.Parameter()

    def output(self):
        return self.make_target(self.dst, processor=waluigi.PyTorchPickleFileProcessor())

    def run(self):
        shutil.move(self.src, self.dst)
        print('Moved output dir: {} -> {}'.format(self.src, self.dst))


class CopyOutput(gokart.TaskOnKart):
    dst = luigi.Parameter(significant=False)
    task = luigi.TaskParameter()

    def output(self):
        return self.make_target(self.dst, processor=waluigi.PyTorchPickleFileProcessor())

    def run(self):
        src = self.task.output().path()
        task_name = self.task.task_family
        shutil.copytree(src, self.dst)
        print('Moved output dir of "{}":{} -> {}'.format(task_name, src, self.dst))


class ZipTaskBase(waluigi.TaskBase):
    configs_overwrite: list = ListParameter()
    names_option: list = ListParameter()
    run_only = ListParameter(None)

    def requires(self):
        assert len(self.configs_overwrite) == len(self.names_option)
        assert len(set(self.names_option)) == len(self.names_option), "names_option must be unique"
        if self.run_only is None:
            run_only = list(range(len(self.configs_overwrite)))
        else:
            run_only = self.run_only

        tasks = {}
        for i, (config, name_option) in enumerate(zip(self.configs_overwrite, self.names_option)):
            if i in run_only:
                # overwrite parent config
                config_updated = recursive_update(self.recursively_make_dict(deepcopy(self.param_kwargs)), dict(**config, name=name_option))
                tasks[name_option] = self.clone_parent(**config_updated)
        return tasks

    def run(self):
        self.dump(self.load())

    def merge_results(self, results):
        unique_keys = set(itertools.chain(*[d.keys() for d in results]))
        output = {k: {} for k in unique_keys}
        for key in unique_keys:
            for mode, result in zip(self.names_option, results):
                if key in result:
                    output[key][mode] = result[key]
        return output
