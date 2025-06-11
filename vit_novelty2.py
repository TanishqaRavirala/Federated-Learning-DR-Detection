# -*- coding: utf-8 -*-
"""train_federated.py - Federated learning with adaptive DP noise and malicious client detection"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import os
import kagglehub
import timm
import flwr as fl
from flwr.common import Metrics, Context, parameters_to_ndarrays, NDArrays
import warnings
import logging
import sys
from typing import List, Tuple, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on {DEVICE} using PyTorch {torch.__version__}, Flower {fl.__version__}, timm {timm.__version__}")

# Number of clients and rounds
NUM_CLIENTS = 5
NUM_ROUNDS = 15  # Increased for better convergence

# Adaptive DP parameters
TRUST_SCORES = [1.0, 0.95, 0.975, 0.925, 0.8]
BASE_NOISE_SCALE = 0.002

# Transformations for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = kagglehub.dataset_download("sovitrath/diabetic-retinopathy-224x224-gaussian-filtered")
dataset_root = f"{dataset_path}/gaussian_filtered_images/gaussian_filtered_images"
dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)

# Model Definition
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.vit = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 11),
            nn.ReLU(),
            nn.BatchNorm1d(11),
            nn.Linear(11, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(net, trainloader, epochs=5, verbose=True):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected in training")
                sys.stdout.flush()
                return False
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")
        sys.stdout.flush()
    return True

# Testing function
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.error("NaN/Inf in test outputs")
                sys.stdout.flush()
                return float('inf'), 0.0
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total
    accuracy = correct / total
    logger.info(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    sys.stdout.flush()
    return loss, accuracy

# Data splitter
BATCH_SIZE = 16

def data_splitter(dataset, num_clients=NUM_CLIENTS):
    datasets_individual = []
    train_datasets_individual = []
    test_datasets_individual = []
    
    for i in range(5):
        classes = torch.tensor([i])
        indices = (torch.tensor(dataset.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        data = torch.utils.data.Subset(dataset, indices)
        datasets_individual.append(data)

    for i in range(5):
        train_temp, test_temp = random_split(
            datasets_individual[i],
            [int(0.8 * len(datasets_individual[i].indices)), len(datasets_individual[i].indices) - int(0.8 * len(datasets_individual[i].indices))],
            generator=torch.Generator().manual_seed(42)
        )
        train_datasets_individual.append(train_temp)
        test_datasets_individual.append(test_temp)

    testset = torch.utils.data.ConcatDataset(test_datasets_individual)
    
    train_datasets = []
    for i in range(5):
        lengths = [int(0.2 * len(train_datasets_individual[i]))] * num_clients
        lengths[-1] = len(train_datasets_individual[i]) - sum(lengths[:-1])
        train_data = random_split(
            train_datasets_individual[i],
            lengths,
            generator=torch.Generator().manual_seed(42)
        )
        train_datasets.append(train_data)

    client_datasets = [torch.utils.data.ConcatDataset([train_datasets[j][i] for j in range(5)]) for i in range(num_clients)]
    trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True) for ds in client_datasets]
    
    test_len = len(testset) // num_clients
    test_lens = [test_len] * num_clients
    test_lens[-1] = len(testset) - test_len * (num_clients - 1)
    valset = random_split(testset, test_lens, generator=torch.Generator().manual_seed(42))
    valloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True) for ds in valset]
    
    return trainloaders, valloaders

# Global train-test split
def split_dataset_for_global_training(dataset):
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    trainset, testset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True), DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Adaptive DP noise
def add_adaptive_dp_noise(parameters, trust_score, base_noise_scale):
    logger.info(f"Adding DP noise with trust_score={trust_score}, base_noise_scale={base_noise_scale}")
    sys.stdout.flush()
    noise_scale = base_noise_scale / max(trust_score, 0.1)
    noisy_params = []
    for param in parameters:
        norm = torch.norm(param).item()
        norm = norm if norm > 0 else 1.0
        noise = torch.normal(0, noise_scale * norm, param.shape, device=param.device)
        noisy_param = param + noise
        noisy_param = torch.clamp(noisy_param, -1.0, 1.0)
        if torch.isnan(noisy_param).any() or torch.isinf(noisy_param).any():
            logger.error(f"NaN/Inf in noisy parameters, norm={norm:.4f}, noise_scale={noise_scale:.4f}")
            sys.stdout.flush()
            return None
        param_norm = torch.norm(noisy_param).item()
        param_mean = torch.mean(noisy_param).item()
        param_std = torch.std(noisy_param).item()
        logger.info(f"Parameter stats: norm={param_norm:.4f}, mean={param_mean:.4f}, std={param_std:.4f}, noise_scale={noise_scale:.4f}")
        sys.stdout.flush()
        noisy_params.append(noisy_param)
    return noisy_params

# Malicious parameter generation
def generate_malicious_parameters(parameters):
    logger.info("Generating malicious parameters")
    sys.stdout.flush()
    malicious_params = []
    for param in parameters:
        malicious_param = torch.randn_like(param) * 50.0
        param_norm = torch.norm(malicious_param).item()
        param_mean = torch.mean(malicious_param).item()
        param_std = torch.std(malicious_param).item()
        logger.info(f"Malicious param stats: norm={param_norm:.4f}, mean={param_mean:.4f}, std={param_std:.4f}")
        sys.stdout.flush()
        malicious_params.append(malicious_param)
    return malicious_params

# Federated learning components
def get_parameters(net):
    return [val.cpu().numpy() for val in net.state_dict().values()]

def set_parameters(net, parameters):
    logger.info("Setting parameters")
    sys.stdout.flush()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {}
    for k, v in params_dict:
        try:
            v = torch.from_numpy(v).to(DEVICE)
            if torch.isnan(v).any() or torch.isinf(v).any():
                logger.error(f"NaN/Inf in parameter {k}")
                sys.stdout.flush()
                return False
            param_norm = torch.norm(v).item()
            logger.info(f"Parameter {k} norm: {param_norm:.4f}")
            state_dict[k] = v.reshape(net.state_dict()[k].shape)
        except Exception as e:
            logger.error(f"Error setting parameter {k}: {e}")
            sys.stdout.flush()
            return False
    try:
        net.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.error(f"Failed to load state dict: {e}")
        sys.stdout.flush()
        return False
    logger.info("Parameters set successfully")
    sys.stdout.flush()
    return True

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, cid, context: Context):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.context = context
        self.trust_score = TRUST_SCORES[int(cid)]
        self.is_initial = True
        self.is_malicious = (cid == "4")
        logger.info(f"Client {self.cid} initialized, trust_score={self.trust_score}, is_malicious={self.is_malicious}")
        sys.stdout.flush()

    def get_parameters(self, config):
        logger.info(f"Client {self.cid} getting parameters, is_initial={self.is_initial}")
        sys.stdout.flush()
        try:
            params = [param.clone() for param in self.net.state_dict().values()]
            if self.is_initial:
                logger.info(f"Client {self.cid} returning clean initial parameters")
                self.is_initial = False
                sys.stdout.flush()
                return [param.cpu().numpy() for param in params]
            if self.is_malicious:
                logger.info(f"Client {self.cid} returning malicious parameters")
                sys.stdout.flush()
                noisy_params = generate_malicious_parameters(params)
            else:
                noisy_params = add_adaptive_dp_noise(params, self.trust_score, BASE_NOISE_SCALE)
                if noisy_params is None:
                    logger.error(f"Client {self.cid} failed to add noise, returning clean params")
                    sys.stdout.flush()
                    return [param.cpu().numpy() for param in params]
            param_stats = [(torch.norm(p).item(), torch.mean(p).item(), torch.std(p).item()) for p in noisy_params]
            logger.info(f"Client {self.cid} parameter stats: {[f'norm={n:.4f}, mean={m:.4f}, std={s:.4f}' for n, m, s in param_stats[:3]]}")
            sys.stdout.flush()
            return [param.cpu().numpy() for param in noisy_params]
        except Exception as e:
            logger.error(f"Client {self.cid} get_parameters failed: {e}, returning clean params")
            sys.stdout.flush()
            return [param.cpu().numpy() for param in self.net.state_dict().values()]

    def fit(self, parameters, config):
        logger.info(f"Client {self.cid} starting fit")
        sys.stdout.flush()
        try:
            if not set_parameters(self.net, parameters):
                logger.error(f"Client {self.cid} failed to set parameters")
                sys.stdout.flush()
                return get_parameters(self.net), len(self.trainloader.dataset), {"error": "set_parameters_failed", "cid": self.cid}
            if self.is_malicious:
                logger.info(f"Client {self.cid} skipping training, returning malicious params")
                sys.stdout.flush()
            else:
                if not train(self.net, self.trainloader, epochs=5, verbose=True):
                    logger.error(f"Client {self.cid} training failed")
                    sys.stdout.flush()
                    return get_parameters(self.net), len(self.trainloader.dataset), {"error": "training_failed", "cid": self.cid}
            logger.info(f"Client {self.cid} finished fit")
            sys.stdout.flush()
            params = self.get_parameters(config)
            return params, len(self.trainloader.dataset), {"cid": self.cid}
        except Exception as e:
            logger.error(f"Client {self.cid} fit failed: {e}")
            sys.stdout.flush()
            return get_parameters(self.net), len(self.trainloader.dataset), {"error": str(e), "cid": self.cid}

    def evaluate(self, parameters, config):
        logger.info(f"Client {self.cid} starting evaluate")
        sys.stdout.flush()
        try:
            if not set_parameters(self.net, parameters):
                logger.error(f"Client {self.cid} failed to set parameters")
                sys.stdout.flush()
                return float('inf'), len(self.valloader.dataset), {"accuracy": 0.0, "cid": self.cid}
            loss, accuracy = test(self.net, self.valloader)
            logger.info(f"Client {self.cid} evaluate: loss {loss:.4f}, accuracy {accuracy:.4f}")
            sys.stdout.flush()
            return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "cid": self.cid}
        except Exception as e:
            logger.error(f"Client {self.cid} evaluate failed: {e}")
            sys.stdout.flush()
            return float('inf'), len(self.valloader.dataset), {"accuracy": 0.0, "cid": self.cid}

def client_fn(context: Context):
    partition_id = context.node_config.get("partition-id", 0)
    logger.info(f"Raw partition-id: {partition_id}, type: {type(partition_id)}")
    sys.stdout.flush()
    try:
        partition_id = int(partition_id)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert partition_id to int: {e}")
        sys.stdout.flush()
        partition_id = 0
    cid = str(partition_id % NUM_CLIENTS)
    logger.info(f"Initializing client {cid} with partition-id {partition_id}")
    sys.stdout.flush()
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(net, trainloader, valloader, cid, context).to_client()

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    return {"global_accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0.0}

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.malicious_clients = {}  # Track malicious clients and rounds

    def aggregate_fit(self, server_round, results, failures):
        logger.info(f"Round {server_round}: Aggregating fit results")
        sys.stdout.flush()
        if not results:
            logger.info(f"Round {server_round}: No results for aggregation")
            sys.stdout.flush()
            return None, {}
        
        # Collect parameters, client IDs, and dataset sizes
        param_list = []
        client_ids = []
        dataset_sizes = []
        for client, fit_res in results:
            try:
                cid = fit_res.metrics.get("cid", "unknown")
                params = parameters_to_ndarrays(fit_res.parameters)
                param_list.append(params)
                client_ids.append(cid)
                # Assume dataset size is available from trainloader (approximated here)
                dataset_sizes.append(len(trainloaders[int(cid)].dataset))
                logger.info(f"Round {server_round}: Processing client {cid}, dataset size={dataset_sizes[-1]}")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Round {server_round}: Failed to process parameters from client: {e}")
                sys.stdout.flush()
                continue
        
        if not param_list:
            logger.info(f"Round {server_round}: No valid parameters for aggregation")
            sys.stdout.flush()
            return None, {}
        
        # Compute parameter norms for malicious detection
        norms = []
        for params in param_list:
            norm = 0.0
            for p in params:
                norm += np.sum(p ** 2)
            norm = np.sqrt(norm)
            norms.append(norm)
            logger.info(f"Round {server_round}: Client {client_ids[len(norms)-1]} raw_norm={norm:.4f}")
            sys.stdout.flush()
        
        # Detect malicious updates
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        threshold = mean_norm + 1.5 * std_norm
        valid_params = []
        valid_clients = []
        valid_dataset_sizes = []
        
        for cid, norm, params in zip(client_ids, norms, param_list):
            logger.info(f"Round {server_round}: Client {cid} norm={norm:.4f}, mean={mean_norm:.4f}, std={std_norm:.4f}, threshold={threshold:.4f}")
            sys.stdout.flush()
            if norm > threshold:
                logger.warning(f"Round {server_round}: Detected malicious update from client {cid}, norm={norm:.4f}, threshold={threshold:.4f}")
                sys.stdout.flush()
                if cid not in self.malicious_clients:
                    self.malicious_clients[cid] = []
                self.malicious_clients[cid].append(server_round)
            else:
                valid_params.append(params)
                valid_clients.append(cid)
                valid_dataset_sizes.append(dataset_sizes[client_ids.index(cid)])
                logger.info(f"Round {server_round}: Accepted update from client {cid}, norm={norm:.4f}")
                sys.stdout.flush()
        
        if not valid_params:
            logger.info(f"Round {server_round}: No valid parameters after filtering")
            sys.stdout.flush()
            return None, {}
        
        # Aggregate valid parameters with weighted averaging
        total_samples = sum(valid_dataset_sizes)
        aggregated_params = []
        for i in range(len(valid_params[0])):
            layer_params = [params[i] for params in valid_params]
            weights = [size / total_samples for size in valid_dataset_sizes]
            agg_param = sum(w * np.array(p, dtype=np.float32) for w, p in zip(weights, layer_params))
            aggregated_params.append(agg_param)
            logger.info(f"Round {server_round}: Aggregated param {i} shape={agg_param.shape}, norm={np.sqrt(np.sum(agg_param**2)):.4f}")
            sys.stdout.flush()
        
        # Save aggregated model
        net = Net().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), aggregated_params)
        state_dict = {}
        for k, v in params_dict:
            try:
                v = np.array(v, dtype=np.float32)
                tensor = torch.from_numpy(v).to(DEVICE)
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logger.error(f"Round {server_round}: NaN/Inf in parameter {k}")
                    sys.stdout.flush()
                    return None, {}
                state_dict[k] = tensor.reshape(net.state_dict()[k].shape)
            except Exception as e:
                logger.error(f"Round {server_round}: Failed to process parameter {k}: {e}")
                sys.stdout.flush()
                return None, {}
        
        try:
            net.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Round {server_round}: Failed to load state dict: {e}")
            sys.stdout.flush()
            return None, {}
        
        torch.save(state_dict, f"model_round_{server_round}.pth")
        logger.info(f"Round {server_round}: Saved model_round_{server_round}.pth")
        sys.stdout.flush()
        
        return fl.common.ndarrays_to_parameters(aggregated_params), {}

# Main execution
def main():
    # Clear old models
    for i in range(1, 15):
        model_path = f"model_round_{i}.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Removed old model file: {model_path}")

    # Log GPU memory
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory available: {total_mem:.2f} GB")
    
    # Global test set
    _, val_global = split_dataset_for_global_training(dataset)

    # Start federated learning
    logger.info("\nStarting federated learning...")
    global trainloaders, valloaders
    trainloaders, valloaders = data_splitter(dataset)
    logger.info("Client data splits:")
    for i in range(NUM_CLIENTS):
        logger.info(f"Client {i} - Train Size: {len(trainloaders[i].dataset)}, Val Size: {len(valloaders[i].dataset)}")

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    client_resources = {"num_gpus": 0.2, "num_cpus": 2}
    os.environ["RAY_DEDUP_LOGS"] = "0"
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"num_gpus": 1, "num_cpus": NUM_CLIENTS * 2, "log_to_driver": False}
    )

    # Print malicious client summary
    logger.info("\nMalicious Client Detection Summary:")
    if strategy.malicious_clients:
        for cid, rounds in strategy.malicious_clients.items():
            logger.info(f"Malicious client detected: Client {cid} in rounds {rounds}")
    else:
        logger.info("No malicious clients detected.")
    sys.stdout.flush()

    # Evaluate final model
    logger.info("\nEvaluating final model...")
    net = Net().to(DEVICE)
    latest_model = None
    for i in range(NUM_ROUNDS, 0, -1):
        if os.path.exists(f"model_round_{i}.pth"):
            latest_model = f"model_round_{i}.pth"
            break
    if latest_model:
        logger.info(f"Loading model: {latest_model}")
        state_dict = torch.load(latest_model, map_location=DEVICE)
        net.load_state_dict(state_dict)
        loss, accuracy = test(net, val_global)
        logger.info(f"Final Test Loss: {loss:.4f}, Final Test Accuracy: {accuracy:.4f}")
    else:
        logger.error("No model files found")

if __name__ == "__main__":
    main()