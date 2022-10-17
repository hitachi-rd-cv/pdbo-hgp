# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
from luigi.execution_summary import _summary_dict, _summary_wrap, _tasks_status, _create_one_line_summary, _group_tasks_by_name_and_status, _get_comments, _ORDERED_STATUSES, _PENDING_SUB_STATUSES, _get_external_workers, \
    execution_summary, _ranging_attributes, _get_str_one_parameter, _get_set_of_params, _get_unique_param_keys, _get_str_ranging_multiple_parameters


class MyLuigiRunResult:
    """
    The result of a call to build/run when passing the detailed_summary=True argument.

    Attributes:
        - one_line_summary (str): One line summary of the progress.
        - summary_text (str): Detailed summary of the progress.
        - status (LuigiStatusCode): Luigi Status Code. See :class:`~luigi.execution_summary.LuigiStatusCode` for what these codes mean.
        - worker (luigi.worker.worker): Worker object. See :class:`~luigi.worker.worker`.
        - scheduling_succeeded (bool): Boolean which is *True* if all the tasks were scheduled without errors.

    """

    def __init__(self, worker, worker_add_run_status=True):
        self.worker = worker
        summary_dict = _summary_dict(worker)
        self.summary_text = _summary_wrap(_my_summary_format(summary_dict, worker))
        self.status = _tasks_status(summary_dict)
        self.one_line_summary = _create_one_line_summary(self.status)
        self.scheduling_succeeded = worker_add_run_status


def _group_tasks_by_task_id_and_status(task_dict):
    """
    Takes a dictionary with sets of tasks grouped by their status and
    returns a dictionary with dictionaries with an array of tasks grouped by
    their status and task name
    """
    group_status = {}
    for task in task_dict:
        if task.task_id not in group_status:
            group_status[task.task_id] = []
        group_status[task.task_id].append(task)
    return group_status


def _my_summary_format(set_tasks, worker):
    group_tasks = {}
    for status, task_dict in set_tasks.items():
        group_tasks[status] = _group_tasks_by_task_id_and_status(task_dict)
    comments = _get_comments(group_tasks)
    num_all_tasks = sum([len(set_tasks["already_done"]),
                         len(set_tasks["completed"]), len(set_tasks["failed"]),
                         len(set_tasks["scheduling_error"]),
                         len(set_tasks["still_pending_ext"]),
                         len(set_tasks["still_pending_not_ext"])])
    str_output = ''
    str_output += 'Scheduled {0} tasks of which:\n'.format(num_all_tasks)
    for status in _ORDERED_STATUSES:
        if status not in comments:
            continue
        str_output += '{0}'.format(comments[status])
        if status != 'still_pending':
            str_output += '{0}\n'.format(_my_get_str(group_tasks[status], status in _PENDING_SUB_STATUSES))
    ext_workers = _get_external_workers(worker)
    group_tasks_ext_workers = {}
    for ext_worker, task_dict in ext_workers.items():
        group_tasks_ext_workers[ext_worker] = _group_tasks_by_name_and_status(task_dict)
    if len(ext_workers) > 0:
        str_output += "\nThe other workers were:\n"
        count = 0
        for ext_worker, task_dict in ext_workers.items():
            if count > 3 and count < len(ext_workers) - 1:
                str_output += "    and {0} other workers".format(len(ext_workers) - count)
                break
            str_output += "    - {0} ran {1} tasks\n".format(ext_worker, len(task_dict))
            count += 1
        str_output += '\n'
    if num_all_tasks == sum([len(set_tasks["already_done"]),
                             len(set_tasks["scheduling_error"]),
                             len(set_tasks["still_pending_ext"]),
                             len(set_tasks["still_pending_not_ext"])]):
        if len(ext_workers) == 0:
            str_output += '\n'
        str_output += 'Did not run any tasks'
    one_line_summary = _create_one_line_summary(_tasks_status(set_tasks))
    str_output += "\n{0}".format(one_line_summary)
    if num_all_tasks == 0:
        str_output = 'Did not schedule any tasks'
    return str_output


def _my_get_str(task_dict, extra_indent):
    """
    This returns a string for each status
    """
    summary_length = execution_summary(summary_length=100).summary_length

    lines = []
    task_names = sorted(task_dict.keys())
    for task_family in task_names:
        tasks = task_dict[task_family]
        tasks = sorted(tasks, key=lambda x: str(x))
        prefix_size = 8 if extra_indent else 4
        prefix = ' ' * prefix_size

        line = None

        if summary_length > 0 and len(lines) >= summary_length:
            line = prefix + "..."
            lines.append(line)
            break
        if len(tasks[0].get_params()) == 0:
            line = prefix + '- {0} {1}()'.format(len(tasks), str(task_family))
        # elif _get_len_of_params(tasks[0]) > 60 or len(str(tasks[0])) > 200 or \
        #         (len(tasks) == 2 and len(tasks[0].get_params()) > 1 and (_get_len_of_params(tasks[0]) > 40 or len(str(tasks[0])) > 100)):
        #     """
        #     This is to make sure that there is no really long task in the output
        #     """
        #     line = prefix + '- {0} {1}(...)'.format(len(tasks), task_family)
        elif len((tasks[0].get_params())) == 1:
            attributes = {getattr(task, tasks[0].get_params()[0][0]) for task in tasks}
            param_class = tasks[0].get_params()[0][1]
            first, last = _ranging_attributes(attributes, param_class)
            if first is not None and last is not None and len(attributes) > 3:
                param_str = '{0}...{1}'.format(param_class.serialize(first), param_class.serialize(last))
            else:
                param_str = '{0}'.format(_get_str_one_parameter(tasks))
            line = prefix + '- {0} {1}({2}={3})'.format(len(tasks), task_family, tasks[0].get_params()[0][0], param_str)
        else:
            ranging = False
            params = _get_set_of_params(tasks)
            unique_param_keys = list(_get_unique_param_keys(params))
            if len(unique_param_keys) == 1:
                unique_param, = unique_param_keys
                attributes = params[unique_param]
                param_class = unique_param[1]
                first, last = _ranging_attributes(attributes, param_class)
                if first is not None and last is not None and len(attributes) > 2:
                    ranging = True
                    line = prefix + '- {0} {1}({2}'.format(len(tasks), task_family, _get_str_ranging_multiple_parameters(first, last, tasks, unique_param))
            if not ranging:
                if len(tasks) == 1:
                    line = prefix + '- {0} {1}'.format(len(tasks), tasks[0])
                if len(tasks) == 2:
                    line = prefix + '- {0} {1} and {2}'.format(len(tasks), tasks[0], tasks[1])
                if len(tasks) > 2:
                    line = prefix + '- {0} {1} ...'.format(len(tasks), tasks[0])
        lines.append(line)
    return '\n'.join(lines)
