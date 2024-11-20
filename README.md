#  Federated Learning Project with Flower

## Overview
This project demonstrates a federated learning setup using Flower and PyTorch. The goal is to train a model in a federated setting with a client-server architecture. The dataset used includes a client dataset and a server test set. Before running the experiments, ensure the data and environment are properly configured as described below.

---

## Prerequisites
- **Python**: Version 3.8 or higher
- **Python Libraries**: Install required dependencies via `requirements.txt`:
  ```bash
  pip install -r requirements.txt
ted Learning Project README

## Overview
This project demonstrates a federated learning setup using Flower and PyTorch. The goal is to train a model in a federated setting with a client-server architecture. The dataset used includes a client dataset and a server test set. Before running the experiments, ensure the data and environment are properly configured as described below.

---

## Prerequisites
- **Python**: Version 3.8 or higher
- **Python Libraries**: Install required dependencies via `requirements.txt`:
  ```bash
  pip install -r requirements.txt
    ```

---

## Data Requirements

### Initial Data Files
- **Client Dataset**: `client_dataset.csv`
- **Server Test Set**: `server_testset.csv`

Both files should be placed in the `/data` directory.

### Label Distribution in `client_dataset.csv`
- **Number of Label 0**: 86,937  
- **Number of Label 1**: 1,635  

If the label distribution does not match these values, update the partitioning logic in the `task.load_or_partition_client_data` function to ensure the data is correctly distributed.

## Instructions

### 1. Clean Up Previous Outputs
Before running the experiment, delete the file `label_proportions.txt`:
```bash
rm label_proportions.txt
```

### 2. Prepare Data
Ensure the following:
- The data files (`client_dataset.csv` and `server_testset.csv`) are present in the `/data` directory.
- The `client_dataset.csv` file adheres to the specified label distribution.
This file is used to store the label distribution in the client dataset. It is initialed in the `client_app.py file` and updated in the `task.py` file.

---
## Notes
- Always delete the `label_proportions.txt` file before running a new experiment to avoid conflicts with previous results.
- If the label distribution in `client_dataset.csv` does not match the specified numbers, modify the logic in the `task.load_or_partition_client_data` function.
- Refer to the comments in the code for additional implementation details.
