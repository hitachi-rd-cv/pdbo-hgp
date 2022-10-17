# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
import traceback

import torch.multiprocessing as multiprocessing
from luigi.event import Event

multiprocessing.set_start_method('spawn', force=True)

from luigi.worker import DequeQueue, SingleProcessPool, TaskException, check_complete, TaskStatusReporter, \
    ContextManagedTaskProcess, Worker


class MyWorker(Worker):
    """
    Worker object communicates with a scheduler.

    Simple class that talks to a scheduler and:

    * tells the scheduler what it has to do + its dependencies
    * asks for stuff to do (pulls it in a loop and runs it)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_result_queue = multiprocessing.Queue()

    def add(self, task, multiprocess=False, processes=0):
        """
        Add a Task for the worker to check and possibly schedule and run.

        Returns True if task and its dependencies were successfully scheduled or completed before.
        """
        if self._first_task is None and hasattr(task, 'task_id'):
            self._first_task = task.task_id
        self.add_succeeded = True
        if multiprocess:
            queue = multiprocessing.Manager().Queue()
            pool = multiprocessing.Pool(processes=processes if processes > 0 else None)
        else:
            queue = DequeQueue()
            pool = SingleProcessPool()
        self._validate_task(task)
        pool.apply_async(check_complete, [task, queue])

        # we track queue size ourselves because len(queue) won't work for multiprocessing
        queue_size = 1
        try:
            seen = {task.task_id}
            while queue_size:
                current = queue.get()
                queue_size -= 1
                item, is_complete = current
                for next in self._add(item, is_complete):
                    if next.task_id not in seen:
                        self._validate_task(next)
                        seen.add(next.task_id)
                        pool.apply_async(check_complete, [next, queue])
                        queue_size += 1
        except (KeyboardInterrupt, TaskException):
            raise
        except Exception as ex:
            self.add_succeeded = False
            formatted_traceback = traceback.format_exc()
            self._log_unexpected_error(task)
            task.trigger_event(Event.BROKEN_TASK, task, ex)
            self._email_unexpected_error(task, formatted_traceback)
            raise
        finally:
            pool.close()
            pool.join()
        return self.add_succeeded

    def _create_task_process(self, task):
        message_queue = multiprocessing.Queue() if task.accepts_messages else None
        reporter = TaskStatusReporter(self._scheduler, task.task_id, self._id, message_queue)
        use_multiprocessing = self._config.force_multiprocessing or bool(self.worker_processes > 1)
        return ContextManagedTaskProcess(
            self._config.task_process_context,
            task, self._id, self._task_result_queue, reporter,
            use_multiprocessing=use_multiprocessing,
            worker_timeout=self._config.timeout,
            check_unfulfilled_deps=self._config.check_unfulfilled_deps,
            check_complete_on_run=self._config.check_complete_on_run,
        )
