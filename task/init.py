import luigi
from luigi import IntParameter, Parameter, DictParameter, BoolParameter
from luigi.util import requires

from constants import KeysTarget
from lib_common import waluigi
from method_task.build_module import build_nodes, build_graph, update_kwargs_build
from method_task.make_datasets import make_datasets


class MakeDatasets(waluigi.TaskBase):
    option_dataset = luigi.DictParameter()
    n_nodes = luigi.IntParameter()
    name_dataset = luigi.Parameter()
    seed_dataset = luigi.IntParameter(None)

    def run(self):
        if self.seed_dataset is None:
            seed = self.fix_random_seed_value
        else:
            seed = self.seed_dataset

        datasets_train, datasets_valid, datasets_test = make_datasets(
            option_dataset=self.option_dataset,
            name_dataset=self.name_dataset,
            n_nodes=self.n_nodes,
            seed=seed,
        )

        self.dump({
            KeysTarget.TRAIN: datasets_train,
            KeysTarget.VALID: datasets_valid,
            KeysTarget.TEST: datasets_test,
        })



class BuildGraph(waluigi.TaskBase):
    n_nodes: int = IntParameter()
    kwargs_init_graph: dict = DictParameter()
    mode_graph: str = Parameter()
    use_cuda: bool = BoolParameter()

    def run(self):
        state_dict = build_graph(
            n_nodes=self.n_nodes,
            mode_graph=self.mode_graph,
            kwargs_init_graph=self.kwargs_init_graph,
            use_cuda=self.use_cuda,
        )

        self.dump({
            KeysTarget.STATE_DICT_GRAPH: state_dict,
        })


@requires(MakeDatasets, BuildGraph)
class MakeKwargsBuildClients(waluigi.TaskBase):
    kwargs_build_base: dict = DictParameter()

    def run(self):
        d_dataset, d_graph = self.load()
        kwargs_build_nodes = update_kwargs_build(
            n_nodes=self.n_nodes,
            datasets_train=d_dataset[KeysTarget.TRAIN],
            kwargs_build_base=self.recursively_make_dict(self.kwargs_build_base),
            mode_graph=self.mode_graph,
            state_dict_graph=d_graph[KeysTarget.STATE_DICT_GRAPH],
        )
        self.dump(kwargs_build_nodes)


@requires(MakeKwargsBuildClients)
class BuildClients(waluigi.TaskBase):
    n_nodes: int = IntParameter()
    name_model: str = Parameter()
    use_cuda: bool = BoolParameter()
    kwargs_init_hparams = DictParameter()

    def run(self):
        kwargs_build_nodes = self.load()
        state_dicts = build_nodes(
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=kwargs_build_nodes,
            use_cuda=self.use_cuda,
            kwargs_init_hparams=self.kwargs_init_hparams,
            seed=self.fix_random_seed_value,
        )

        self.dump({
            KeysTarget.STATE_DICTS: state_dicts,
        })


@requires(MakeKwargsBuildClients)
class BuildClientsWOHyperParams(waluigi.TaskBase):
    n_nodes: int = IntParameter()
    name_model: str = Parameter()
    use_cuda: bool = BoolParameter()

    def run(self):
        kwargs_build_nodes = self.load()
        state_dicts = build_nodes(
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=kwargs_build_nodes,
            use_cuda=self.use_cuda,
            seed=self.fix_random_seed_value,
        )

        self.dump({
            KeysTarget.STATE_DICTS: state_dicts,
        })

