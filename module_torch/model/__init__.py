from constants import AbbrModels
from module_torch.model.cnn_cifar10 import CNNCIFAR10
from module_torch.model.cnn_emnist import CNNEMNIST
from module_torch.model.cnn_mnist import CNNMNIST
from module_torch.model.fc_mnist import FCMNIST
from module_torch.model.learners_ensemble import MultiLearners, MultiLanguageLearners
from module_torch.model.lr_mnist import LogisticRegressionMNIST
from module_torch.model.lr_toy import LogisticRegressionToy
from module_torch.model.lstm_shakespeare import NextCharacterLSTM
from module_torch.model.mn_cifar10 import MobileNetCIFAR10
from module_torch.model.mn_cifar100 import MobileNetCIFAR100

D_MODELS = {
    AbbrModels.LEARNERS_ENSEMBLE: MultiLearners,
    AbbrModels.LANGUAGE_LEARNERS_ENSEMBLE: MultiLanguageLearners,
    AbbrModels.CNN_EMNIST: CNNEMNIST,
    AbbrModels.CNN_CIFAR10: CNNCIFAR10,
    AbbrModels.LR_TOY: LogisticRegressionToy,
    AbbrModels.MN_CIFAR10: MobileNetCIFAR10,
    AbbrModels.MN_CIFAR100: MobileNetCIFAR100,
    AbbrModels.LSTM_SHAKESPEARE: NextCharacterLSTM,
    AbbrModels.FC_MNIST: FCMNIST,
    AbbrModels.LR_MNIST: LogisticRegressionMNIST,
    AbbrModels.CNN_MNIST: CNNMNIST,
}
