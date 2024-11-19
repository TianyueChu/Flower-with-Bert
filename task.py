"""quickstart-compose: A Flower / PyTorch app."""
import logging
# Set logging level to DEBUG
# logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, \
    DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizer

import fcntl
import ast
import datasets
from typing import List, Tuple
from torch.utils.data import DataLoader
from flwr.common import Metrics


from logging import WARNING
from typing import Callable, Optional, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg,aggregate_inplace

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedCustom(Strategy):
    """ Customized Federated Averaging strategy with server-side evaluation

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    server_testset_csv_path: Path to the CSV file containing server-side evaluation data.

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        server_testset_csv_path: str
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.server_eval_data = load_server_data(server_testset_csv_path)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        initial_parameters = [param.detach().cpu().numpy() for param in model.parameters()]
        # Convert to Flower-compatible Parameters object
        initial_parameters = ndarrays_to_parameters(initial_parameters)
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.latest_aggregated_parameters = parameters_aggregated

        # Optionally evaluate the global model after aggregation
        eval_loss = None
        if server_round % 1 == 0:  # Run evaluation every round (you can adjust frequency)
            print(f"Evaluating global model at the end of round {server_round}")
            eval_loss = self.evaluate_global_model(server_round)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate_global_model(self, server_round: int) -> float:
        """Evaluate BERT model with aggregated parameters."""

        # Load the aggregated parameters
        if hasattr(self, 'latest_aggregated_parameters'):
            # Assume parameters_aggregated is a dict of layer weights
            model = load_parameters_to_bert(self.latest_aggregated_parameters)

        tokenizer =  DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(f"Evaluating model in Server Round {server_round}...")
        eval_results = test(model, self.server_eval_data)

        return float(eval_results['eval_loss'])

class IntrusionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(client_id):
    # client_df = pd.read_csv(f'/app/data/client_dataset_{client_id}.csv')
    client_df = pd.read_csv(f'./data/client_dataset_{client_id}.csv')
    X = client_df['log_line']
    y = client_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # Initialize the tokenizer once and use it for all tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=100)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=100)

    trainset = IntrusionDataset(train_encodings, y_train.tolist())
    testset = IntrusionDataset(test_encodings, y_test.tolist())

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples, tokenizer


def train(model, train_dataset, epochs=1):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/',
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/',
        use_cpu=not torch.cuda.is_available(),
    )
    # Use a DataCollator to handle padding and ensure batches are of uniform length
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    return trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'prec': prec,
        'rec': rec,
        'f1': f1
    }


def test(model, test_dataset):
    eval_args = TrainingArguments(
        output_dir=f'./results/',
        per_device_eval_batch_size=32,
        do_eval=True,  # Explicitly set to indicate this is evaluation-only
        use_cpu=not torch.cuda.is_available()  # Ensures device compatibility
    )
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Initialize Trainer for evaluation only
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Print evaluation results
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    return eval_results


def get_unique_proportion_from_file(proportions_file_path):
    """Retrieve and remove the first label proportion from the file for a unique assignment."""
    with open(proportions_file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file for safe read-modify-write

        # Read and parse the current label proportions from the file
        proportions_str = f.read().strip()
        current_proportions = ast.literal_eval(proportions_str)

        # Get the first proportion and update the list
        if current_proportions:
            current_proportion = current_proportions.pop(0)  # Retrieve and remove the first element
        else:
            raise ValueError("No more label proportions available")

        # Rewind the file, write the updated list, and truncate the file
        f.seek(0)
        f.write(str(current_proportions))
        f.truncate()

        fcntl.flock(f, fcntl.LOCK_UN)  # Release the file lock

    return current_proportion


def load_or_partition_client_data(client_id, label_proportions_file_path,label_column='label', label_0_total=86937, label_1_total=1635):
    """
    Loads the client data from an existing CSV file if available; otherwise, partitions the data for the client.

    Args:
    - client_id (int): Unique identifier for the client.
    - label_proportions_file_path (str): Path to the file containing label proportions.
    - label_column (str): Name of the label column in the data.
    - label_0_total (int): Total count of label 0 in the data.
    - label_1_total (int): Total count of label 1 in the data.

    Returns:
    - DataFrame: Partitioned client data.
    """

    # Define paths
    dataset_path = './data/client_dataset.csv'
    client_file_path = f'./data/client_dataset_{client_id}.csv'

    # Check if the client's CSV file already exists
    if os.path.exists(client_file_path):
        # If file exists, load and return data
        logging.debug(f"Loading existing data for client ID: {client_id}")
        client_data = pd.read_csv(client_file_path)
        return client_data

    # If file does not exist, proceed with partitioning
    logging.debug(f"Client data for ID {client_id} does not exist. Partitioning data.")

    # Get a unique label proportion for this client from the file or list
    current_proportion = get_unique_proportion_from_file(label_proportions_file_path)  # You can define this function as needed
    print(f"Assigning label proportion {current_proportion} to client ID {client_id}")

    # Load the main dataset
    df = pd.read_csv(dataset_path)

    # Separate data by labels
    label_0_data = df[df[label_column] == 0]
    label_1_data = df[df[label_column] == 1]

    # Calculate sample sizes based on the fixed total counts and current proportions
    label_0_size = min(int(label_0_total * current_proportion[0]), len(label_0_data))
    label_1_size = min(int(label_1_total * current_proportion[1]), len(label_1_data))

    # Sample from each label
    part_label_0 = label_0_data.sample(n=label_0_size, replace=False, random_state=42)
    part_label_1 = label_1_data.sample(n=label_1_size, replace=False, random_state=42)

    # Concatenate and shuffle the part
    client_data = pd.concat([part_label_0, part_label_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save partitioned data to CSV for future reuse
    client_data.to_csv(client_file_path, index=False)

    df.drop(part_label_0.index, inplace=True)
    df.drop(part_label_1.index, inplace=True)

    df.to_csv(dataset_path, index=False)
    return client_data

def partition(client_id, partition_size):
    print(f"Running partition for client_id={client_id}")

    # Ensure the directory exists (though likely redundant with volume mounts)
    # print("Ensuring data directory exists at /app/data")

    # Define paths
    dataset_path = './data/client_dataset.csv'
    client_file_path = f'./data/client_dataset_{client_id}.csv'
    print(f"Dataset path: {dataset_path}")
    print(f"Client file path: {client_file_path}")

    # Check if the client's dataset already exists
    if not os.path.exists(client_file_path):
        if not os.path.exists(dataset_path):
            print(f"Error: The file '{dataset_path}' was not found.")
            raise FileNotFoundError(f"The file '{dataset_path}' was not found.")

        print("Reading dataset")
        dataset = pd.read_csv(dataset_path)

        # Check if the partition size is valid
        if partition_size > len(dataset):
            raise ValueError("Partition size is larger than the available data.")

        # Take the partition for the client and reset indices
        client_partition = dataset.iloc[:partition_size].reset_index(drop=True)

        # Save client partition
        try:
            client_partition.to_csv(client_file_path, index=False)
            print(f"Saved client dataset for client {client_id} at {client_file_path}")
        except Exception as e:
            print(f"Failed to save client dataset for client {client_id}. Error: {e}")

        # Save remaining dataset
        remaining_dataset = dataset.iloc[partition_size:].reset_index(drop=True)
        try:
            remaining_dataset.to_csv(dataset_path, index=False)
            print(f"Saved remaining dataset back to {dataset_path}")
        except Exception as e:
            print(f"Failed to save the remaining dataset. Error: {e}")

def load_server_data(csv_path: str):
    """Load server-side evaluation data from a CSV file."""
    print(f"Loading server-side data from: {csv_path}")
    try:
        # Read CSV data
        df = pd.read_csv(csv_path)
        X = df['log_line']
        y = df['label']
        print("Server data loaded successfully.")

        # Initialize the tokenizer once and use it for all tokenization
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        test_encodings = tokenizer(X.tolist(), truncation=True, padding=True, max_length=100)
        testset = IntrusionDataset(test_encodings, y.tolist())
        return testset

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading server data: {e}")
        raise


def load_parameters_to_bert(parameters_aggregated):
    """Loads aggregated parameters into BERT model."""
    print("Loading parameters into BERT model")
    # Initialize BERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    ndarrays = parameters_to_ndarrays(parameters_aggregated)

    # Iterate over model layers and assign parameters
    state_dict = {}
    for (name, param), ndarray in zip(model.named_parameters(), ndarrays):
        state_dict[name] = torch.tensor(ndarray, dtype=param.dtype)
        print(f"Loaded parameter for layer: {name}")

    # Load updated parameters into BERT model
    model.load_state_dict(state_dict,strict=True)
    print("Parameters loaded successfully into BERT model.")
    return model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    for m in metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def dataset_loader(client_id):
    # Load data
    data = pd.read_csv(f"./data/client_dataset_{client_id}.csv")

    features = data['log_line'].values.tolist()
    labels = data['label'].values.tolist()

    ds_dict = {'feature': features, 'label': labels}
    ds = datasets.Dataset.from_dict(ds_dict)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenization(example):
        return tokenizer(
            example['feature'], padding=True,
            truncation=True, max_length=100,
        )

    ds = ds.map(tokenization, batched=True)
    ds = ds.remove_columns(['feature'])
    ds = ds.train_test_split(test_size=0.01, seed=42)
    train_ds = ds['train']
    train_ds.save_to_disk(f"./data/train_data/client_dataset_{client_id}")
    test_ds = ds['test']
    test_ds.save_to_disk(f"./data/test_data/client_dataset_{client_id}")

    num_examples = {
        "trainset": len(train_ds),
        "testset": len(test_ds)
    }
    return num_examples

