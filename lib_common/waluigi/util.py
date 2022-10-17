# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
class inherits:
    """
    Task inheritance.

    *New after Luigi 2.7.6:* multiple arguments support.

    Usage:

    .. code-block:: python

        class AnotherTask(luigi.Task):
            m = luigi.IntParameter()

        class YetAnotherTask(luigi.Task):
            n = luigi.IntParameter()

        @inherits(AnotherTask)
        class MyFirstTask(luigi.Task):
            def requires(self):
               return self.clone_parent()

            def run(self):
               print self.m # this will be defined
               # ...

        @inherits(AnotherTask, YetAnotherTask)
        class MySecondTask(luigi.Task):
            def requires(self):
               return self.clone_parents()

            def run(self):
               print self.n # this will be defined
               # ...
    """

    def __init__(self, *tasks_to_inherit, names_non_inherited_param=None):
        super(inherits, self).__init__()
        if not tasks_to_inherit:
            raise TypeError("tasks_to_inherit cannot be empty")

        self.tasks_to_inherit = tasks_to_inherit
        self.names_non_inherited_param = names_non_inherited_param or []

    def __call__(self, task_that_inherits):
        # Get all parameter objects from each of the underlying tasks
        for task_to_inherit in self.tasks_to_inherit:
            for param_name, param_obj in task_to_inherit.get_params():
                # Check if the parameter exists in the inheriting task
                if not hasattr(task_that_inherits, param_name):
                    if not param_name in self.names_non_inherited_param:
                        # If not, add it to the inheriting task
                        setattr(task_that_inherits, param_name, param_obj)
                else:
                    assert param_name not in self.names_non_inherited_param, f'{param_name} is already attributed.'

        # Modify task_that_inherits by adding methods
        def clone_parent(_self, **kwargs):
            return _self.clone(cls=self.tasks_to_inherit[0], **kwargs)

        task_that_inherits.clone_parent = clone_parent

        def clone_parents(_self, **kwargs):
            return [
                _self.clone(cls=task_to_inherit, **kwargs)
                for task_to_inherit in self.tasks_to_inherit
            ]

        task_that_inherits.clone_parents = clone_parents

        return task_that_inherits


class requires:
    """
    Same as :class:`~luigi.util.inherits`, but also auto-defines the requires method.

    *New after Luigi 2.7.6:* multiple arguments support.

    """

    def __init__(self, *tasks_to_require, names_non_inherited_param=None):
        super(requires, self).__init__()
        if not tasks_to_require:
            raise TypeError("tasks_to_require cannot be empty")

        self.tasks_to_require = tasks_to_require
        self.names_non_inherited_param = names_non_inherited_param

    def __call__(self, task_that_requires):
        task_that_requires = inherits(*self.tasks_to_require, names_non_inherited_param=self.names_non_inherited_param)(task_that_requires)

        # Modify task_that_requires by adding requires method.
        # If only one task is required, this single task is returned.
        # Otherwise, list of tasks is returned
        def requires(_self):
            return _self.clone_parent() if len(self.tasks_to_require) == 1 else _self.clone_parents()

        task_that_requires.requires = requires

        return task_that_requires
