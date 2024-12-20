"""quickstart-compose: A Flower / PyTorch app."""
import logging
import time
import hashlib
import torch
from transformers import DistilBertForSequenceClassification
from collections import OrderedDict
from pathlib import Path
from datasets import load_from_disk

import flwr
from flwr.client import Client

from flwr.client import ClientApp
from flwr.common import Context
from task import  test, train, load_or_partition_client_data,dataset_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {DEVICE}")

current_time = str(time.time()).encode()  # Get the current timestamp as a string
client_id_global = hashlib.md5(current_time).hexdigest()

# Path to the shared label proportions file in ./configs directory
proportions_file_path = Path("./configs/label_proportions.txt")

# Ensure the ./configs directory exists
proportions_file_path.parent.mkdir(parents=True, exist_ok=True)

# Initialize label proportions in the file if it doesn’t exist
initial_proportions = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
if not proportions_file_path.exists():
    with open(proportions_file_path, 'w') as f:
        f.write(str(initial_proportions))


class BertClient(flwr.client.NumPyClient):
    def __init__(self, net, trainset, testset, num_examples):
        logging.debug("BertClient initialized with model, trainset, testset")
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.num_examples = num_examples


    def get_parameters(self, config):
        # Get model parameters as a list of numpy arrays
        logging.debug("Getting model parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of numpy arrays
        logging.debug("Setting model parameters")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        logging.debug("Model parameters set successfully")

    def fit(self, parameters, config):
        # Set parameters and train the model
        logging.debug("Entering fit function")
        self.set_parameters(parameters)
        logging.debug("Parameters set, starting training")
        train(
            model=self.net,
            train_dataset=self.trainset,
            epochs=1
        )
        logging.debug("Training completed")
        self.evaluate(self.get_parameters(config={}), config)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        # Set parameters
        logging.debug("Entering evaluate function")
        self.set_parameters(parameters)
        logging.debug("Parameters set, starting evaluation")
        eval_results = test(self.net, self.testset)
        logging.debug(f"Evaluation completed with results: {eval_results}")
        return float(eval_results['eval_loss']), self.num_examples["testset"], {
            "accuracy": float(eval_results['eval_accuracy'])}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    logging.debug("Starting client_fn to initialize Flower client")

    # Load model
    logging.debug("Loading DistilBERT model for sequence classification")
    net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    logging.debug("Model loaded successfully")

    client_id = context.node_id
    if client_id == -1:
        client_id = client_id_global
    print(f"Client ID: {client_id}")

    # Partition data for this client
    logging.debug(f"Partitioning data for client ID: {client_id}")
    load_or_partition_client_data(client_id, proportions_file_path)
    logging.debug("Data partitioned successfully")

    # Load data
    logging.debug("Loading data for client")
    num_examples = dataset_loader(client_id)

    train_path = f"data/train_data/client_dataset_{client_id}"
    test_path = f"data/test_data/client_dataset_{client_id}"

    # Load datasets
    trainset = load_from_disk(train_path)
    testset= load_from_disk(test_path)
    logging.debug("Data loaded successfully")

    # Create and return a Flower client
    logging.debug("Creating BertClient instance")
    return BertClient(net, trainset, testset, num_examples).to_client()


# Create the ClientApp
logging.debug("Starting ClientApp")
app = ClientApp(client_fn=client_fn)
logging.debug("ClientApp created successfully")

flwr.client.start_client(server_address="127.0.0.1:8080", client_fn=client_fn, grpc_max_message_length=1_073_741_824)