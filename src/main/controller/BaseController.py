from src.main.service import BaseService


class BaseController:

    def __init__(self,batch_size, path_dataset,path_checkpoint, path_model_params):
        self.path_dataset = path_dataset
        self.path_checkpoint = path_checkpoint
        self.path_model_params = path_model_params
        self.service = BaseService


    def trainer(self, epochs, name_experiment, is_test):
        self.service.trainer(
            self.path_checkpoint,
            self.path_model_params,
            epochs,
            name_experiment,
            is_test
        )