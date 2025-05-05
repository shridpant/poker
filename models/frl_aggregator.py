import copy
import torch
import numpy as np

class FRLAggregator:
    """
    Implements federated aggregation methods for model parameters or gradients.
    
    Supports multiple aggregation strategies:
    - FedAvg: Standard weighted averaging
    - Median: Robust coordinate-wise median (Byzantine-resistant)
    - Trimmed Mean: Robust trimmed mean aggregation
    """
    def __init__(self, robust=False, agg_method="median"):
        self.robust = robust
        self.agg_method = agg_method if robust else "median"
        
    def aggregate_parameters(self, parameters_list, weights=None):
        """
        Aggregate model parameters from multiple agents
        
        Args:
            parameters_list: List of model parameters from each agent
            weights: Optional weights for each agent's contribution
        """
        if not parameters_list:
            return None
            
        if weights is None:
            weights = [1.0 / len(parameters_list) for _ in range(len(parameters_list))]
            
        if self.agg_method == "fedavg":
            return self._fedavg(parameters_list, weights)
        elif self.agg_method == "median":
            return self._robust_median(parameters_list)
        elif self.agg_method == "trimmed_mean":
            return self._robust_trimmed_mean(parameters_list)
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")
    
    def _fedavg(self, parameters_list, weights):
        """Standard FedAvg aggregation"""
        aggregated_params = copy.deepcopy(parameters_list[0])
        for param_idx, param in enumerate(aggregated_params):
            param.zero_()
            for client_idx, client_params in enumerate(parameters_list):
                param.add_(client_params[param_idx] * weights[client_idx])
        return aggregated_params
    
    def _robust_median(self, parameters_list):
        """Robust aggregation using coordinate-wise median"""
        aggregated_params = copy.deepcopy(parameters_list[0])
        for param_idx, _ in enumerate(aggregated_params):
            # Stacking parameters from all clients for this layer
            stacked_params = torch.stack([client_params[param_idx] for client_params in parameters_list])
            # Calculating median across clients (dimension 0)
            median_update = torch.median(stacked_params, dim=0).values
            aggregated_params[param_idx].copy_(median_update)
        return aggregated_params
    
    def _robust_trimmed_mean(self, parameters_list, trim_ratio=0.2):
        """Robust aggregation using trimmed mean"""
        aggregated_params = copy.deepcopy(parameters_list[0])
        n_clients = len(parameters_list)
        n_trim = int(n_clients * trim_ratio)
        
        for param_idx, _ in enumerate(aggregated_params):
            # Stacking parameters from all clients for this layer
            stacked_params = torch.stack([client_params[param_idx] for client_params in parameters_list])
            # Sorting values across clients
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            # Trimming the smallest and largest values
            if n_trim > 0 and n_clients > 2*n_trim:
                trimmed = sorted_params[n_trim:-n_trim]
            else:
                trimmed = sorted_params
            # Calculating mean of remaining values
            mean_update = torch.mean(trimmed, dim=0)
            aggregated_params[param_idx].copy_(mean_update)
        return aggregated_params
    
    def aggregate_gradients(self, gradients_list, weights=None):
        """
        Aggregate gradients from multiple agents
        
        Args:
            gradients_list: List of gradients from each agent
            weights: Optional weights for each agent's contribution
        """
        # Using the same aggregation logic as for parameters
        return self.aggregate_parameters(gradients_list, weights)
