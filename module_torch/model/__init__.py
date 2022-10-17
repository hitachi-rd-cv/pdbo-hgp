from constants import AbbrModels
from module_torch.model.cnn_emnist import CNNEMNIST
from module_torch.model.learners_ensemble import MultiLearners
from module_torch.model.lr_toy import LogisticRegressionToy

D_MODELS = {
    AbbrModels.LEARNERS_ENSEMBLE: MultiLearners,
    AbbrModels.CNN_EMNIST: CNNEMNIST,
    AbbrModels.LR_TOY: LogisticRegressionToy,
}
