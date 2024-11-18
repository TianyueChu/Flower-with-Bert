"""quickstart-compose: A Flower / PyTorch app."""
import flwr as fl
import logging
# Set logging level to DEBUG
# logging.basicConfig(level=logging.DEBUG)
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.backends.opt_einsum import strategy

# from quickstart_compose.task import FedCustom, weighted_average
from task import FedCustom, weighted_average

# Configure logging
# logging.basicConfig(filename="server.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training

    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)


# Create FedAvg strategy
strategy = FedCustom(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=5,  # Sample 100% of available clients for evaluation
    min_fit_clients=5,  # Never sample less than 5 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=5,  # Wait until all 5 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
    server_testset_csv_path="./data/server_testset.csv"
)

# strategy = FedAvg()

# Create ServerApp
# app = ServerApp(server_fn=server_fn)

fl.server.start_server(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=3), strategy=strategy, grpc_max_message_length=1_073_741_824)
