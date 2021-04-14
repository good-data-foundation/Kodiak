import numpy as np

import torch
from torch.utils.data import DataLoader

from goodDataML.mpc.mpc_status import MPCServerStatus
import goodDataML.connection.proto.mpc_message_pb2 as mpc_message
from goodDataML.learning.horizontal.model_base import BaseMmodel
from goodDataML.learning.utils.pytorch_utils import send_data_to_device

from typing import Dict, List, Any


class PytorchModel(BaseMmodel):
    """
    Model training class for Pytorch
    """

    def __init__(self, server_status, query_uuid, config):
        # type: (MPCServerStatus, str, Dict[str, Any]) -> None
        super(PytorchModel, self).__init__()
        self.config = config['model_config']
        self.server_status = server_status
        self.query_uuid = query_uuid
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TODO add parallel training and distributed training

    def fit(self, data):
        # type: (DataLoader) -> None
        """
        Fit the model with the training data

        :param data: A torch data loader, it contains both training features and labels.
        :return:
        """

        epoch = self.config['model_config']['epoch']
        optimizer = self.config['model_config']['optimizer']
        loss_fn = self.config['model_train_config']['loss_fn']

        # send model to gpu/cpu
        self.model.to(self.device)
        for i in range(epoch):
            for batch in data:
                # send data to gpu/cpi
                batch = send_data_to_device(batch, self.device)
                # the default set up is the first element in the batch is features
                # the second element is label
                train_data = batch[0]
                target = batch[1]

                optimizer.zero_grad()
                output = self.model(train_data)
                loss = loss_fn(output, target)

                # TODO add loss to log
                loss.backward()
                optimizer.step()

                # send parameters to MPC node and get global update
                self.mpc_update_param(len(target), i)

    def predict(self, data):
        # type: (DataLoader) -> List[np.array]
        """
        Make the prediction with trained model

        :param data: A torch data loader, it contains both prediction features.
        :return:
        """
        all_predction = []
        self.model.eval()
        # send model to gpu/cpu
        self.model.to(self.device)
        for batch in data:
            # send data to gpu/cpi
            batch = send_data_to_device(batch, self.device)
            # the default set up is the first element in the batch is features
            # the second element is label
            train_data = batch[0]
            output = self.model(train_data)
            # if the GPU training is enabled, use cpu() to convert the data from GPU to CPU.
            # if the CPU training is enabled, cpu() won't do any work
            all_predction.append(output.cpu().data)
        return all_predction

    def evaluate(self, data, evaluate_metrics):
        # type: (DataLoader, Dict[str, Any]) -> Dict[str, np.array]
        """
        Evaluate the model performance

        :param data: A torch data loader, it contains both training features and labels.
        :param evaluate_metrics: A dictionary of multiple evaluation metric functions.
        :return:
        """
        eval_results = {}
        self.model.eval()
        self.model.to(self.device)
        for batch in data:
            # send data to gpu/cpi
            batch = send_data_to_device(batch, self.device)
            # the default set up is the first element in the batch is features
            # the second element is label
            train_data = batch[0]
            target = batch[1]
            output = self.model(train_data)
            for metric_name in evaluate_metrics.keys():
                # TODO convert output and target to numpy
                # if the GPU training is enabled, use cpu() to convert the data from GPU to CPU.
                # if the CPU training is enabled, cpu() won't do any work
                output = output.cpu().data
                metric = evaluate_metrics[metric_name](output, target)

                # TODO check the predict from model is nd array or 1d array
                if metric_name not in eval_results:
                    eval_results[metric_name] = metric
                else:
                    eval_results[metric_name] = np.append(eval_results[metric_name], metric)

        return eval_results

    def load_model(self, model_dict):
        # type: (str) -> None
        """
        Load entire Pytorch model
        :param model_dict: model path
        :return:
        """
        self.model = torch.load(model_dict)

    def save_model(self, model_dict):
        # type: (str) -> None
        """
        Save entire Pytroch model
        :param model_dict: model path
        :return:
        """
        torch.save(self.model, model_dict)

    def mpc_update_param(self, data_size, epoch):
        # type: (int, int) -> None
        """
        Current only send one copy to one MPC, need to add secret sharing function to
        break parameters in multiple parts.
        :param data_size:
        :param epoch:
        :return:
        """

        # get current parameters
        param = list(self.model.parameters())

        # wrap it to grpc message
        request1 = mpc_message.AggregationRequest(agg_type=mpc_message.AggregationRequest.WEIGHT_AVG,
                                                  gradients=param, data_size=data_size)

        # send to MPC and wait for response
        self.server_status.submit_gradients(request1)
        new_param = self.server_status.wait_or_aggregate(self.query_uuid, epoch)

        # assign the new param
        self.model.load_state_dict(new_param)
