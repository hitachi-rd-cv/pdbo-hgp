# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
import logging
import signal

from luigi import lock
from luigi.interface import _WorkerSchedulerFactory, core, PidLockAlreadyTakenExit
from luigi.setup_logging import InterfaceLogging

from lib_common.waluigi.execution_summary import MyLuigiRunResult
from lib_common.waluigi.worker import MyWorker


def mybuild(tasks, worker_scheduler_factory=None, detailed_summary=False, **env_params):
    """
    Run internally, bypassing the cmdline parsing.

    Useful if you have some luigi code that you want to run internally.
    Example:

    .. code-block:: python

        luigi.build([MyTask1(), MyTask2()], local_scheduler=True)

    One notable difference is that `build` defaults to not using
    the identical process lock. Otherwise, `build` would only be
    callable once from each process.

    :param tasks:
    :param worker_scheduler_factory:
    :param env_params:
    :return: True if there were no scheduling errors, even if tasks may fail.
    """
    if "no_lock" not in env_params:
        env_params["no_lock"] = True

    luigi_run_result = _my_schedule_and_run(tasks, worker_scheduler_factory, override_defaults=env_params)
    return luigi_run_result if detailed_summary else luigi_run_result.scheduling_succeeded


def _my_schedule_and_run(tasks, worker_scheduler_factory=None, override_defaults=None):
    """
    :param tasks:
    :param worker_scheduler_factory:
    :param override_defaults:
    :return: True if all tasks and their dependencies were successfully run (or already completed);
             False if any error occurred. It will return a detailed response of type LuigiRunResult
             instead of a boolean if detailed_summary=True.
    """

    if worker_scheduler_factory is None:
        worker_scheduler_factory = _MyWorkerSchedulerFactory()
    if override_defaults is None:
        override_defaults = {}
    env_params = core(**override_defaults)

    InterfaceLogging.setup(env_params)

    kill_signal = signal.SIGUSR1 if env_params.take_lock else None
    if (not env_params.no_lock and
            not (lock.acquire_for(env_params.lock_pid_dir, env_params.lock_size, kill_signal))):
        raise PidLockAlreadyTakenExit()

    if env_params.local_scheduler:
        sch = worker_scheduler_factory.create_local_scheduler()
    else:
        if env_params.scheduler_url != '':
            url = env_params.scheduler_url
        else:
            url = 'http://{host}:{port:d}/'.format(
                host=env_params.scheduler_host,
                port=env_params.scheduler_port,
            )
        sch = worker_scheduler_factory.create_remote_scheduler(url=url)

    worker = worker_scheduler_factory.create_worker(
        scheduler=sch, worker_processes=env_params.workers, assistant=env_params.assistant)

    success = True
    logger = logging.getLogger('luigi-interface')
    with worker:
        for t in tasks:
            success &= worker.add(t, env_params.parallel_scheduling, env_params.parallel_scheduling_processes)
        logger.info('Done scheduling tasks')
        success &= worker.run()
    luigi_run_result = MyLuigiRunResult(worker, success)
    logger.info(luigi_run_result.summary_text)
    return luigi_run_result


class _MyWorkerSchedulerFactory(_WorkerSchedulerFactory):
    def create_worker(self, scheduler, worker_processes, assistant=False):
        return MyWorker(
            scheduler=scheduler, worker_processes=worker_processes, assistant=assistant)
