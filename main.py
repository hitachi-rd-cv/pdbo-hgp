import re
import sys

from dotenv import load_dotenv
from luigi.cmdline_parser import CmdlineParser
from luigi.freezing import FrozenOrderedDict

from lib_common.waluigi import print_tree, pop_cmndline_arg, MyListParameter
from task import *


def get_named_params(task_cls, default_named_params, cp, ignore_unmatch_args=False):
    names_task_param = [x[0] for x in task_cls.get_params()]
    params_task = {k: v for k, v in default_named_params.items() if k in names_task_param}
    for k, v in cp._get_task_kwargs().items():
        if not k in names_task_param:
            if ignore_unmatch_args:
                pass
            else:
                raise ValueError(f'Invalid argument "{k}"')
        else:
            if isinstance(v, FrozenOrderedDict):
                if len(v) > 0:
                    recursive_update(params_task[k], v)
                else:
                    params_task[k] = v
            else:
                params_task[k] = v
    return params_task



if __name__ == '__main__':
    load_dotenv()
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    path_config = sys.argv.pop(2)
    if '-m' in sys.argv:
        conf_func_name = pop_cmndline_arg('-m')
    else:
        conf_func_name = 'get_config'

    exec(f'from {path_config} import {conf_func_name}')
    all_params = eval(f'{conf_func_name}()')

    if '-s' in sys.argv:
        pop_cmndline_arg('-s', flag=True)
        load_dotenv()
        d_mongo = dict(
            db_name=os.getenv('MONGO_DB'),
            mongo_auth=os.getenv('MONGO_AUTH'),
        )
    else:
        d_mongo = dict()

    # by adding --tree arg, you can see the dependencies of the tasks and the status of Done/Pendings of them
    # when this arg set, the task will not run.
    if '--tree' in sys.argv:
        pop_cmndline_arg('--tree', flag=True)
        cmdline_args = sys.argv[1:]
        with CmdlineParser.global_instance(cmdline_args) as cp:
            task_cls = cp._get_task_cls()
            params_target_task = get_named_params(task_cls, all_params, cp)
            task_target = task_cls(**params_target_task)
            print(print_tree(task_target))
        sys.exit()

    # by adding `--removes {task_class1},{task_class2}` it removes all the intermediate files between {task_class1} ({task_class2}) and the target task you specified at the beginning of the shell command.
    if '--removes' in sys.argv or '--only-removes' in sys.argv:
        if '--removes' in sys.argv:
            rm_arg = '--removes'
        elif '--only-removes' in sys.argv:
            rm_arg = '--only-removes'
        else:
            raise ValueError(sys.argv)
        names_removed_task_unparsed = pop_cmndline_arg(rm_arg)
        names_removed_task_parsed = MyListParameter.parse(names_removed_task_unparsed)
        with CmdlineParser.global_instance(sys.argv[1:]) as cp:
            task_cls = cp._get_task_cls()
            params_target_task = get_named_params(task_cls, all_params, cp)
            task_target = task_cls(**params_target_task)

            params_removing_task = get_named_params(MoveOutputs, all_params, cp, ignore_unmatch_args=True)
            task_remove = MoveOutputs(target_task=task_target, removes=names_removed_task_parsed, **params_removing_task)
            result_remove = luigi.build([task_remove], local_scheduler=True)
            assert result_remove, f'{task_remove} failed.'

        if rm_arg == '--only-removes':
            sys.exit()

    with CmdlineParser.global_instance(sys.argv[1:]) as cp:
        task_cls = cp._get_task_cls()
        params_target_task = get_named_params(task_cls, all_params, cp)
        task_target = task_cls(**params_target_task, **d_mongo)
        luigi.build([task_target])
