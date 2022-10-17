# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Sacred (https://github.com/IDSIA/sacred), Copyright (c) 2014 Klaus Greff.
# The portions of the following codes are licensed under the MIT licence.
# The full license text is available at (https://github.com/IDSIA/sacred/blob/master/LICENSE.txt).
# ------------------------------------------------------------------------

import datetime
import os
import os.path
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stdout
from copy import copy
from datetime import timedelta
from tempfile import NamedTemporaryFile

import sacred
import wrapt
from sacred import commandline_options, SETTINGS
from sacred.config.custom_containers import fallback_dict
from sacred.config.signature import Signature
from sacred.experiment import gather_command_line_options
from sacred.host_info import get_host_info
from sacred.initialize import get_config_modifications, get_command, get_configuration, get_scaffolding_and_config_name, \
    gather_ingredients_topological, initialize_logging, distribute_config_updates, distribute_presets, \
    create_scaffolding
from sacred.metrics_logger import MetricsLogger, ScalarMetricLogEntry, opt
from sacred.randomness import set_global_seed, create_rnd, get_seed
from sacred.run import Run
from sacred.stdout_capturing import no_tee, tee_output_python, CapturedStdout, flush
from sacred.utils import apply_backspaces_and_linefeeds, convert_to_nested_dict, iterate_flattened, set_by_dotted_path, recursive_update, SacredInterrupt, ConfigError, optional_kwargs_decorator, join_paths

import torch

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'  # for git error
SETTINGS['CAPTURE_MODE'] = 'sys'

class MyMetricsLogger(MetricsLogger):
    def log_scalar_metric(self, metric_name, value, step=None, time_delta=None):
        """
        Add a new measurement.

        The measurement will be processed by the MongoDB observer
        during a heartbeat event.
        Other observers are not yet supported.

        :param metric_name: The name of the metric, e.g. training.loss.
        :param value: The measured value.
        :param step: The step number (integer), e.g. the iteration number
                    If not specified, an internal counter for each metric
                    is used, incremented by one.
        :param time_delta: datetime.timedelta
        """
        if opt.has_numpy:
            np = opt.np
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(step, np.generic):
                step = step.item()
        if step is None:
            step = self._metric_step_counter.get(metric_name, -1) + 1
        if time_delta is None:
            time = datetime.datetime.utcnow()
        else:
            time = datetime.datetime(1970, 1, 1) + time_delta
        self._logged_metrics.put(
            ScalarMetricLogEntry(metric_name, step, time, value)
        )
        self._metric_step_counter[metric_name] = step


class MyRun(Run):
    def __init__(
            self,
            config,
            config_modifications,
            main_function,
            observers,
            root_logger,
            run_logger,
            experiment_info,
            host_info,
            pre_run_hooks,
            post_run_hooks,
            bypassed_config={},
            captured_out_filter=None,
    ):
        self._id = None
        """The ID of this run as assigned by the first observer"""

        self.captured_out = ""
        """Captured stdout and stderr"""

        self.config = config
        """The final configuration used for this run"""

        self.config_modifications = config_modifications
        """A ConfigSummary object with information about config changes"""

        self.experiment_info = experiment_info
        """A dictionary with information about the experiment"""

        self.host_info = host_info
        """A dictionary with information about the host"""

        self.info = {}
        """Custom info dict that will be sent to the observers"""

        self.root_logger = root_logger
        """The root logger that was used to create all the others"""

        self.run_logger = run_logger
        """The logger that is used for this run"""

        self.main_function = main_function
        """The main function that is executed with this run"""

        self.observers = observers
        """A list of all observers that observe this run"""

        self.pre_run_hooks = pre_run_hooks
        """List of pre-run hooks (captured functions called before this run)"""

        self.post_run_hooks = post_run_hooks
        """List of post-run hooks (captured functions called after this run)"""

        self.result = None
        """The return value of the main function"""

        self.status = None
        """The current status of the run, from QUEUED to COMPLETED"""

        self.start_time = None
        """The datetime when this run was started"""

        self.stop_time = None
        """The datetime when this run stopped"""

        self.debug = False
        """Determines whether this run is executed in debug mode"""

        self.pdb = False
        """If true the pdb debugger is automatically started after a failure"""

        self.meta_info = {}
        """A custom comment for this run"""

        self.beat_interval = 10.0  # sec
        """The time between two heartbeat events measured in seconds"""

        self.unobserved = False
        """Indicates whether this run should be unobserved"""

        self.force = False
        """Disable warnings about suspicious changes"""

        self.queue_only = False
        """If true then this run will only fire the queued_event and quit"""

        self.captured_out_filter = captured_out_filter
        """Filter function to be applied to captured output"""

        self.fail_trace = None
        """A stacktrace, in case the run failed"""

        self.capture_mode = None
        """Determines the way the stdout/stderr are captured"""

        self._heartbeat = None
        self._failed_observers = []
        self._output_file = None

        self._metrics = MyMetricsLogger()
        self.bypassed_config = bypassed_config

    def __call__(self, *args):
        r"""Start this run.

        Parameters
        ----------
        \*args
            parameters passed to the main function

        Returns
        -------
            the return value of the main function

        """
        if self.start_time is not None:
            raise RuntimeError(
                "A run can only be started once. "
                "(Last start was {})".format(self.start_time)
            )

        if self.unobserved:
            self.observers = []
        else:
            self.observers = sorted(self.observers, key=lambda x: -x.priority)

        self.warn_if_unobserved()
        set_global_seed(self.config["seed"])

        if self.capture_mode is None and not self.observers:
            capture_mode = "no"
        else:
            capture_mode = self.capture_mode
        capture_mode, capture_stdout = get_stdcapturer(capture_mode)
        self.run_logger.debug('Using capture mode "%s"', capture_mode)

        if self.queue_only:
            self._emit_queued()
            return
        try:
            with capture_stdout() as self._output_file:
                self._emit_started()
                self._start_heartbeat()
                self._execute_pre_run_hooks()
                result_return = self.main_function(**self.bypassed_config)
                self.result = None  # change: returning result can omniboard very slow
                self._execute_post_run_hooks()
                if self.result is not None:
                    self.run_logger.info("Result: {}".format(self.result))
                elapsed_time = self._stop_time()
                self.run_logger.info("Completed after %s", elapsed_time)
                self._get_captured_output()
            self._stop_heartbeat()
            self._emit_completed(self.result)
        except (SacredInterrupt, KeyboardInterrupt) as e:
            self._stop_heartbeat()
            status = getattr(e, "STATUS", "INTERRUPTED")
            self._emit_interrupted(status)
            raise
        except BaseException:
            exc_type, exc_value, trace = sys.exc_info()
            self._stop_heartbeat()
            self._emit_failed(exc_type, exc_value, trace.tb_next)
            raise
        finally:
            self._warn_about_failed_observers()
            self._wait_for_observers()

        return result_return

    def add_artifact(self, filename, name=None, metadata=None, content_type=None):
        with redirect_stdout(open(os.devnull, 'w')):
            super().add_artifact(filename, name, metadata, content_type)

    def log_scalar(self, metric_name, value, step=None, time_delta=None):
        assert not isinstance(value, torch.Tensor), f'Tensor is not available but value={value} type: {type(value)}'
        self._metrics.log_scalar_metric(metric_name, value, step, time_delta)


class Experiment(sacred.Experiment):
    def __init__(
            self,
            name=None,
            ingredients=(),
            interactive=False,
            base_dir=None,
            additional_host_info=None,
            additional_cli_options=None,
            save_git_info=False,
            mongo_auth=None,
            db_name='no_name',
            use_file_storage_server=True
    ):
        '''
        automatically append observers and replace self.captured_out_filter with modified apply_backspaces_and_linefeeds
        to fixes the problem around stdout of tqdm progress bar

        Args:d
            name:
            ingredients:
            interactive:
            base_dir:
            additional_host_info:
            additional_cli_options:
            save_git_info:
            use_mongo:
        '''
        super().__init__(name, ingredients, interactive, base_dir, additional_host_info,
                         additional_cli_options, save_git_info)

        if mongo_auth is not None:
            url = 'mongodb://{}/?authMechanism=SCRAM-SHA-1'.format(mongo_auth)
            self.observers.append(sacred.observers.MongoObserver(url=url, db_name=db_name))

        if use_file_storage_server:
            self.observers.append(sacred.observers.FileStorageObserver(os.path.join('.sacred_runs', db_name)))

        # add python modules to resources
        # for path in glob('**/*.py', recursive=True):
        #     self.add_source_file(path)

        # it fixes stdout problem
        self.captured_out_filter = apply_backspaces_and_linefeeds

    def main(self, function):
        """
        Decorator to define the main function of the experiment.

        The main function of an experiment is the default command that is being
        run when no command is specified, or when calling the run() method.

        Usually it is more convenient to use ``automain`` instead.
        """
        captured = self.command(function)
        self.default_command = captured.__name__
        return captured

    def _create_run(
            self,
            command_name=None,
            config_updates=None,
            named_configs=(),
            info=None,
            meta_info=None,
            options=None,
            bypassed_config={},
    ):
        command_name = command_name or self.default_command
        if command_name is None:
            raise RuntimeError(
                "No command found to be run. Specify a command "
                "or define a main function."
            )

        default_options = self.get_default_options()
        if options:
            default_options.update(options)
        options = default_options

        # call option hooks
        for oh in self.option_hooks:
            oh(options=options)

        run = create_run(
            self,
            command_name,
            config_updates,
            named_configs=named_configs,
            force=options.get(commandline_options.force_option.get_flag(), False),
            log_level=options.get(commandline_options.loglevel_option.get_flag(), None),
            bypassed_config=bypassed_config
        )
        if info is not None:
            run.info.update(info)

        run.meta_info["command"] = command_name
        run.meta_info["options"] = options

        if meta_info:
            run.meta_info.update(meta_info)

        options_list = gather_command_line_options() + self.additional_cli_options
        for option in options_list:
            option_value = options.get(option.get_flag(), False)
            if option_value:
                option.apply(option_value, run)

        self.current_run = run

        return run

    @optional_kwargs_decorator
    def capture(self, function=None, prefix=None):
        """
        Decorator to turn a function into a captured function.

        The missing arguments of captured functions are automatically filled
        from the configuration if possible.
        See :ref:`captured_functions` for more information.

        If a ``prefix`` is specified, the search for suitable
        entries is performed in the corresponding subtree of the configuration.
        """
        if function in self.captured_functions:
            return function
        captured_function = create_captured_function(function, prefix=prefix)
        self.captured_functions.append(captured_function)
        return captured_function

    @optional_kwargs_decorator
    def command(self, function=None, prefix=None, unobserved=False):
        """
        Decorator to define a new command for this Ingredient or Experiment.

        The name of the command will be the name of the function. It can be
        called from the command-line or by using the run_command function.

        Commands are automatically also captured functions.

        The command can be given a prefix, to restrict its configuration space
        to a subtree. (see ``capture`` for more information)

        A command can be made unobserved (i.e. ignoring all observers) by
        passing the unobserved=True keyword argument.
        """
        captured_f = self.capture(function, prefix=prefix)
        captured_f.unobserved = unobserved
        self.commands[function.__name__] = captured_f
        return captured_f

@contextmanager
def tee_output_fd():
    """Duplicate stdout and stderr to a file on the file descriptor level."""
    with NamedTemporaryFile(mode="w+", newline='') as target:
        # with NamedTemporaryFile(mode="w+", newline='') as target:
        original_stdout_fd = 1
        original_stderr_fd = 2
        target_fd = target.fileno()

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        try:
            # start_new_session=True to move process to a new process group
            # this is done to avoid receiving KeyboardInterrupts (see #149)
            tee_stdout = subprocess.Popen(
                ["tee", "-a", target.name],
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=1,
            )
            tee_stderr = subprocess.Popen(
                ["tee", "-a", target.name],
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=2,
            )
        except (FileNotFoundError, OSError, AttributeError):
            # No tee found in this operating system. Trying to use a python
            # implementation of tee. However this is slow and error-prone.
            tee_stdout = subprocess.Popen(
                [sys.executable, "-m", "sacred.pytee"],
                stdin=subprocess.PIPE,
                stderr=target_fd,
            )
            tee_stderr = subprocess.Popen(
                [sys.executable, "-m", "sacred.pytee"],
                stdin=subprocess.PIPE,
                stdout=target_fd,
            )

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)
        out = CapturedStdout(target)

        try:
            yield out  # let the caller do their printing
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            tee_stdout.wait(timeout=1)
            tee_stderr.wait(timeout=1)

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            out.finalize()


def get_stdcapturer(mode=None):
    mode = mode if mode is not None else SETTINGS.CAPTURE_MODE
    capture_options = {"no": no_tee, "fd": tee_output_fd, "sys": tee_output_python}
    if mode not in capture_options:
        raise KeyError(
            "Unknown capture mode '{}'. Available options are {}".format(
                mode, sorted(capture_options.keys())
            )
        )
    return mode, capture_options[mode]


def create_run(
        experiment,
        command_name,
        config_updates=None,
        named_configs=(),
        force=False,
        log_level=None,
        bypassed_config={},
):
    sorted_ingredients = gather_ingredients_topological(experiment)
    scaffolding = create_scaffolding(experiment, sorted_ingredients)
    # get all split non-empty prefixes sorted from deepest to shallowest
    prefixes = sorted(
        [s.split(".") for s in scaffolding if s != ""],
        reverse=True,
        key=lambda p: len(p),
    )

    # --------- configuration process -------------------

    # Phase 1: Config updates
    config_updates = config_updates or {}
    config_updates = convert_to_nested_dict(config_updates)
    root_logger, run_logger = initialize_logging(experiment, scaffolding, log_level)
    distribute_config_updates(prefixes, scaffolding, config_updates)

    # Phase 2: Named Configs
    for ncfg in named_configs:
        scaff, cfg_name = get_scaffolding_and_config_name(ncfg, scaffolding)
        scaff.gather_fallbacks()
        ncfg_updates = scaff.run_named_config(cfg_name)
        distribute_presets(prefixes, scaffolding, ncfg_updates)
        for ncfg_key, value in iterate_flattened(ncfg_updates):
            set_by_dotted_path(config_updates, join_paths(scaff.path, ncfg_key), value)

    distribute_config_updates(prefixes, scaffolding, config_updates)

    # Phase 3: Normal config scopes
    for scaffold in scaffolding.values():
        scaffold.gather_fallbacks()
        scaffold.set_up_config()

        # update global config
        config = get_configuration(scaffolding)
        # run config hooks
        config_hook_updates = scaffold.run_config_hooks(
            config, command_name, run_logger
        )
        recursive_update(scaffold.config, config_hook_updates)

    # Phase 4: finalize seeding
    for scaffold in reversed(list(scaffolding.values())):
        scaffold.set_up_seed()  # partially recursive

    config = get_configuration(scaffolding)
    config_modifications = get_config_modifications(scaffolding)

    # ----------------------------------------------------

    experiment_info = experiment.get_experiment_info()
    host_info = get_host_info(experiment.additional_host_info)
    main_function = get_command(scaffolding, command_name)
    pre_runs = [pr for ing in sorted_ingredients for pr in ing.pre_run_hooks]
    post_runs = [pr for ing in sorted_ingredients for pr in ing.post_run_hooks]

    run = MyRun(
        config,
        config_modifications,
        main_function,
        copy(experiment.observers),
        root_logger,
        run_logger,
        experiment_info,
        host_info,
        pre_runs,
        post_runs,
        captured_out_filter=experiment.captured_out_filter,
        bypassed_config=bypassed_config
    )

    if hasattr(main_function, "unobserved"):
        run.unobserved = main_function.unobserved

    run.force = force

    for scaffold in scaffolding.values():
        scaffold.finalize_initialization(run=run)

    return run


def create_captured_function(function, prefix=None):
    sig = Signature(function)
    function.signature = sig
    function.uses_randomness = "_seed" in sig.arguments or "_rnd" in sig.arguments
    function.logger = None
    function.config = {}
    function.rnd = None
    function.run = None
    function.prefix = prefix
    return captured_function(function)


@wrapt.decorator
def captured_function(wrapped, instance, args, kwargs):
    options = fallback_dict(
        # config is only for recording in my Experiment
        # wrapped.config, _config=wrapped.config, _log=wrapped.logger, _run=wrapped.run
        kwargs, _config=kwargs, _log=wrapped.logger, _run=wrapped.run
    )
    if wrapped.uses_randomness:  # only generate _seed and _rnd if needed
        options["_seed"] = get_seed(wrapped.rnd)
        options["_rnd"] = create_rnd(options["_seed"])

    bound = instance is not None
    args, kwargs = wrapped.signature.construct_arguments(args, kwargs, options, bound)
    if wrapped.logger is not None:
        wrapped.logger.debug("Started")
        start_time = time.time()
    # =================== run actual function =================================
    with ConfigError.track(wrapped.config, wrapped.prefix):
        result = wrapped(*args, **kwargs)
    # =========================================================================
    if wrapped.logger is not None:
        stop_time = time.time()
        elapsed_time = timedelta(seconds=round(stop_time - start_time))
        wrapped.logger.debug("Finished after %s.", elapsed_time)

    return result
