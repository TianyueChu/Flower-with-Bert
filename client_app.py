"""quickstart-compose: A Flower / PyTorch app."""
import logging
logging.basicConfig(level=logging.INFO)
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizer

from collections import OrderedDict

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import load_data, test, train, partition

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {DEVICE}")

client_id_global = int(str(time.time())[6:].replace('.', ''))

class BertClient(fl.client.NumPyClient):
    def __init__(self, net, trainset, testset, num_examples, tokenizer):
        logging.debug("BertClient initialized with model, trainset, testset, and tokenizer")
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.num_examples = num_examples
        self.tokenizer = tokenizer
        logging.debug("BertClient initialized with model, trainset, testset, and tokenizer")

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
        trainer = train(
            model=self.net,
            train_dataset=self.trainset,
            tokenizer=self.tokenizer,
            epochs=1
        )
        logging.debug("Training completed")
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        # Set parameters
        logging.debug("Entering evaluate function")
        self.set_parameters(parameters)
        logging.debug("Parameters set, starting evaluation")
        eval_results = test(self.net, self.testset, self.tokenizer)
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

    # Get client ID from context
    client_id = context.node_id
    if client_id == -1:
        client_id = client_id_global
    logging.debug(f"Client ID: {client_id}")

    # Partition data for this client
    logging.debug(f"Partitioning data for client ID: {client_id}")
    partition(client_id, 100)
    logging.debug("Data partitioned successfully")

    # Load data
    logging.debug("Loading data for client")
    trainset, testset, num_examples, tokenizer = load_data(client_id)
    logging.debug("Data loaded successfully")

    # Create and return a Flower client
    logging.debug("Creating BertClient instance")
    return BertClient(net, trainset, testset, num_examples, tokenizer).to_client()


# Create the ClientApp
logging.debug("Starting ClientApp")
app = ClientApp(client_fn=client_fn)
logging.debug("ClientApp created successfully")

# fl.client.start_client(server_address="81.41.186.137:8081", client_fn=client_fn, grpc_max_message_length=1_073_741_824)

