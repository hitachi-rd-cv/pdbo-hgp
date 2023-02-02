class AbbrModels:
    LEARNERS_ENSEMBLE = 'learners_ensemble'
    CNN_EMNIST = 'cnn_emnist'
    CNN_CIFAR10 = 'cnn_cifar10'
    LR_TOY = 'lr_toy'
    MN_CIFAR10 = 'mn_cifar10'
    MN_CIFAR100 = 'mn_cifar100'
    LSTM_SHAKESPEARE = 'lstm_shakespeare'
    LANGUAGE_LEARNERS_ENSEMBLE = 'language_learners_ensemble'


class NamesEvalMetric:
    LOSS_BARE_MEAN = 'loss_bare_mean'
    LOSS_MEAN = 'loss_mean'
    ACCURACY = 'accuracy'


class NamesHyperLoss:
    L2_REGULARIZER = 'l2_regularizer'



class ModesGraph:
    CONSTANTS = 'constants'
    STOCHASTIC_BIDIRECTED = 'stochastic_bidirected'
    STOCHASTIC_DIRECTED = 'stochastic_directed'
    SELF_LOOP = 'self_loop'
    ERDOS_RENYI = 'graph_erdos_renyi'


class ModesGossip:
    WEIGHT_GIVEN = 'weight_given'
    NORMALIZE = 'normalize'


class KeysTarget:
    HYPER_GRADIENTS_NODES = 'hypergrads_nodes'
    HYPER_GRADIENTS_NODES_STEPS = 'hypergrads_nodes_steps'
    STATE_DICTS = 'state_dicts'
    STATE_DICTS_MODEL = 'state_dicts'
    STATE_DICTS_LEARNERS = 'state_dicts_learners'
    STATE_DICTS_HYPER = 'hyper_state_dicts'
    STATE_DICTS_HYPER_OPTIMIZER = 'hyper_optimizer_state_dicts'
    STATE_DICTS_ESTIMATOR = 'state_dicts_estimator'
    HYPER_STATE_DICTS_OF_HYPER_STEPS = 'hyper_state_dicts_of_hyper_steps'
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    D_D_VAL_EVAL_NODES = 'd_d_val_eval_nodes'
    D_D_VAL_EVAL_MEAN = 'd_d_val_eval_mean'
    D_D_VAL_EVAL_BOTTOM = 'd_d_val_eval_bottom'
    D_D_VAL_EVAL_MEAN_HYPER_STEPS = 'd_d_val_eval_mean_hyper_steps'
    D_D_VAL_EVAL_BOTTOM_HYPER_STEPS = 'd_d_val_eval_bottom_hyper_steps'
    TIME_HYPER_UPDATE = 'time_hyper_update'
    STATE_DICT_GRAPH = 'state_dict_graph'
    LOG_TRAIN = 'log_training'
    DS_METRIC_FEDEM = 'd_metric_fedem'
    DS_METRIC_FEDEM_HYPER_STEPS = 'd_metric_fedem_hyper_steps'
    ERROR_NORM_HYPERGRAD_STEPS = 'error_norm_hypergrad_steps'


class KeysOptionHGP:
    DEPTH = 'depth'
    DUMPING = 'dumping'
    USE_TRUE_EXPECTED_EDGES = 'use_true_expected_edges'
    MODE_UPDATE = 'mode_update'
    ALPHA_V = 'alpha_v'
    ALPHA_W = 'alpha_w'

class ModesHGPUpdate:
    SIMULTANEOUS = 'simultaneous'
    ALT_U_V = 'alt_u_v'
    ALT_V_U = 'alt_v_u'
    U_TO_V = 'u_to_v'



class NamesDataset:
    EMNIST = 'emnist'
    TOY_MNIST = 'toy_mnist'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    SHAKE_SPEARE = 'shakespeare'


class TypesDevice:
    CUDA = 'cuda'
    CPU = 'cpu'


class KeysOptionTrainInsig:
    NAMES_METRIC_LOG = 'names_valid_metric'
    LOG_EVERY = 'log_every'


class KeysOptionTrainSig:
    LR_SCHEDULER = 'lr_scheduler'
    DISABLE_DEBIAS_WEIGHT = 'disable_debias_weight'
    DROP_LAST = 'drop_last'
    KWARGS_LR_SCHEDULER = 'kwargs_lr_scheduler'
    MODE_SGP = 'iteration'


class ModesSGP:
    NEDIC = "nedic"
    ASSRAN = "assran"


class KeysOptionEval:
    NAME = 'name_eval_metric'


class KeysOptionHGPInsig:
    NAMES_METRIC_LOG = 'names_hgp_metric'


class NamesHGPMetric:
    U_NORM = 'u_norm'
    V_NORM = 'v_norm'
    W_NORM = 'w_norm'
    V_DIFF_NORM = 'v_diff_norm'


class NamesHyperParam:
    SOFTMAX_CATEGORY_WEIGHTS = 'softmax_category_weights'
    LEARNERS_WEIGHTS = 'learners_weights'
    LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS = 'learners_weights_and_softmax_category_weights'
    HYPER_EXP_LOGITS_WEIGHTS = 'hyper_exp_logits_weights'
    HYPER_SOFTMAX_LOGITS_WEIGHTS = 'hyper_softmax_logits_weights'
    LEARNERS_WEIGHTS_AND_MULTI_SOFTMAX_LOGITS_WEIGHTS = 'learners_weights_and_multi_softmax_logits_weights'
    LEARNERS_WEIGHTS_AND_SINGLE_SOFTMAX_LOGITS_WEIGHTS = 'learners_weights_and_single_softmax_logits_weights'
    DUMMY = 'dummy'

class NamesLoader:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

class HyperOptimizers:
    SGD = 'sgd'
    ADAM = 'adam'


class NamesLrScheduler:
    CONST = 'const'
    MULTI_STEP = 'multi_step'

