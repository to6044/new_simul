"""
EXP3 Algorithm-based Cell On/Off Scenario for KISS Network Simulator

This module implements an EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
based cell on/off strategy for energy efficiency optimization in 5G networks.
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import dataclasses
from AIMM_simulator import Scenario


class EXP3CellOnOff(Scenario):
    """
    EXP3 algorithm-based cell on/off scenario that dynamically selects which cells to turn off
    to optimize network energy efficiency (throughput/power).
    
    The algorithm maintains weights for each possible combination of cells to turn off (arms)
    and updates them based on the observed rewards (network efficiency).
    """
    
    def __init__(self, sim, k_cells: int, n_cells_off: int, 
                 interval: float = 1.0, delay: float = 0.0,
                 gamma: float = 0.1, warmup_episodes: int = 100,
                 enable_warmup: bool = True, ensure_min_selection: bool = True,
                 linear_power_model: bool = False, power_model_k: float = 0.05,
                 learning_log_file: str = None, final_model_file: str = None,
                 verbose: bool = True):
        """
        Initialize EXP3 Cell On/Off scenario.
        
        Parameters:
        -----------
        sim : Simv2
            The simulation object
        k_cells : int
            Total number of cells in the network
        n_cells_off : int
            Number of cells to turn off simultaneously
        interval : float
            Time interval between decisions (default: 1.0)
        delay : float
            Initial delay before starting the scenario (default: 0.0)
        gamma : float
            Exploration parameter for EXP3 (default: 0.1)
        warmup_episodes : int
            Number of initial episodes for random exploration (default: 100)
        enable_warmup : bool
            Whether to enable warm-up phase (default: True)
        ensure_min_selection : bool
            Whether to ensure each arm is selected at least once (default: True)
        linear_power_model : bool
            Whether to use linear power model based on UE count (default: False)
        power_model_k : float
            Coefficient for linear power model (default: 0.05)
        learning_log_file : str
            Path to save learning progress (default: None)
        final_model_file : str
            Path to save final model (default: None)
        verbose : bool
            Whether to print detailed progress (default: True)
        """
        self.sim = sim
        self.k_cells = k_cells
        self.n_cells_off = n_cells_off
        self.interval = interval
        self.delay = delay
        self.gamma = gamma
        self.warmup_episodes = warmup_episodes if enable_warmup else 0
        self.ensure_min_selection = ensure_min_selection
        self.linear_power_model = linear_power_model
        self.power_model_k = power_model_k
        self.verbose = verbose
        
        # Create log directory if needed
        if learning_log_file:
            os.makedirs(os.path.dirname(learning_log_file), exist_ok=True)
        self.learning_log_file = learning_log_file
        self.final_model_file = final_model_file
        
        # Calculate number of arms (combinations of n cells from k cells)
        from math import comb
        self.n_arms = comb(k_cells, n_cells_off)
        
        # Generate all possible combinations of cells to turn off
        from itertools import combinations
        self.arms = list(combinations(range(k_cells), n_cells_off))
        
        # Initialize EXP3 weights
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms
        
        # Track statistics
        self.episode_count = 0
        self.arm_selection_count = np.zeros(self.n_arms)
        self.cumulative_rewards = np.zeros(self.n_arms)
        self.reward_history = []
        self.weight_history = []
        self.probability_history = []
        self.selected_arm_history = []
        
        # Store baseline performance (all cells on)
        self.baseline_efficiency = None
        self.baseline_throughput = None
        self.baseline_power = None
        
        # Store original cell powers
        self.original_cell_powers = {}
        
        # Cell energy models reference
        self.cell_energy_models = None
        
        if self.verbose:
            print(f"EXP3CellOnOff initialized: {k_cells} cells, turning off {n_cells_off} cells")
            print(f"Total arms (combinations): {self.n_arms}")
            print(f"Exploration parameter (gamma): {gamma}")
            print(f"Warm-up episodes: {self.warmup_episodes if enable_warmup else 'Disabled'}")
        
    def set_cell_energy_models(self, cell_energy_models: Dict):
        """Set reference to cell energy models dictionary"""
        self.cell_energy_models = cell_energy_models
        
    def get_network_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate current network metrics.
        
        Returns:
        --------
        Tuple[float, float, float]
            (total_throughput_Mbps, total_power_kW, efficiency_bps_per_J)
        """
        total_throughput = 0.0
        total_power = 0.0
        active_cells = 0
        
        for cell in self.sim.cells:
            # Skip cells that are turned off
            if cell.power_dBm <= -100:  # Effectively off
                continue
                
            # Get cell throughput
            cell_tp = cell.get_cell_throughput()
            total_throughput += cell_tp
            
            # Get cell power
            if self.cell_energy_models and cell.i in self.cell_energy_models:
                energy_model = self.cell_energy_models[cell.i]
                
                try:
                    # Use linear power model if enabled
                    if self.linear_power_model and hasattr(energy_model, 'get_linear_power_watts'):
                        n_ues = len(cell.attached)
                        cell_power_watts = energy_model.get_linear_power_watts(n_ues, self.power_model_k)
                    else:
                        cell_power_watts = energy_model.get_cell_power_watts(self.sim.env.now)
                    
                    # Validate power value
                    if np.isfinite(cell_power_watts) and cell_power_watts > 0:
                        cell_power_kW = cell_power_watts / 1e3
                        total_power += cell_power_kW
                        active_cells += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to get power for cell {cell.i}: {e}")
                    # Use default power estimation
                    cell_power_watts = 2000  # 2kW default for active cell
                    cell_power_kW = cell_power_watts / 1e3
                    total_power += cell_power_kW
                    active_cells += 1
            else:
                # Fallback: estimate power from power_dBm
                if cell.power_dBm > -100:
                    # Rough estimation: 43 dBm ≈ 2kW total consumption
                    cell_power_kW = 2.0  # Default 2kW per active cell
                    total_power += cell_power_kW
                    active_cells += 1
        
        # Add minimum power to avoid division by zero
        if total_power == 0:
            # All cells might be off or power calculation failed
            # Use number of active cells * minimum power
            total_power = max(active_cells * 1.5, 0.1)  # At least 1.5kW per active cell
        
        # Calculate efficiency (handle division by zero)
        if total_power > 0:
            efficiency = (total_throughput * 1e6) / (total_power * 1e3)  # bits/J
        else:
            efficiency = 0.0
            
        if self.verbose and self.episode_count % 10 == 0:
            print(f"  Debug: Active cells={active_cells}, Total TP={total_throughput:.2f}, Total Power={total_power:.2f}")
            
        return total_throughput, total_power, efficiency
    
    def turn_cells_off(self, cell_indices: List[int]):
        """Turn off specified cells by setting their power to -inf"""
        for idx in cell_indices:
            if idx < len(self.sim.cells):
                cell = self.sim.cells[idx]
                # Store original power if not already stored
                if idx not in self.original_cell_powers:
                    self.original_cell_powers[idx] = cell.power_dBm
                # Turn off cell
                cell.set_power_dBm(-np.inf)
                
    def turn_cells_on(self, cell_indices: List[int]):
        """Turn on specified cells by restoring their original power"""
        for idx in cell_indices:
            if idx < len(self.sim.cells) and idx in self.original_cell_powers:
                cell = self.sim.cells[idx]
                cell.set_power_dBm(self.original_cell_powers[idx])
                
    def select_arm(self) -> int:
        """
        Select an arm (combination of cells to turn off) based on EXP3 algorithm.
        
        Returns:
        --------
        int
            Index of selected arm
        """
        # During warm-up, select randomly
        if self.episode_count < self.warmup_episodes:
            return self.sim.rng.choice(self.n_arms)
        
        # Ensure minimum selection if enabled
        if self.ensure_min_selection:
            # Find arms that haven't been selected yet
            unselected_arms = np.where(self.arm_selection_count == 0)[0]
            if len(unselected_arms) > 0:
                return self.sim.rng.choice(unselected_arms)
        
        # Update probabilities
        self.update_probabilities()
        
        # Select arm based on probabilities
        return self.sim.rng.choice(self.n_arms, p=self.probabilities)
    
    def update_probabilities(self):
        """Update arm selection probabilities based on weights"""
        # EXP3 probability calculation
        sum_weights = np.sum(self.weights)
        exploitation_probs = self.weights / sum_weights
        
        # Mix exploitation with exploration
        self.probabilities = (1 - self.gamma) * exploitation_probs + self.gamma / self.n_arms
        
    def calculate_reward(self, efficiency: float) -> float:
        """
        Calculate normalized reward based on network efficiency.
        
        Parameters:
        -----------
        efficiency : float
            Network efficiency in bits/J
            
        Returns:
        --------
        float
            Normalized reward in [0, 1]
        """
        # Handle zero efficiency
        if efficiency <= 0:
            return 0.0
            
        # Use baseline for normalization if available
        if self.baseline_efficiency is not None and self.baseline_efficiency > 0:
            # Normalize relative to baseline
            reward = efficiency / self.baseline_efficiency
            # Clip to [0, 2] then scale to [0, 1]
            reward = np.clip(reward, 0, 2) / 2
        else:
            # Simple normalization without baseline
            # Assume efficiency typically ranges from 0 to 2e5 bits/J
            # (based on ~300 Mbps throughput and ~30 kW power)
            typical_efficiency = 1e4  # 10,000 bits/J as a reasonable baseline
            reward = np.clip(efficiency / typical_efficiency, 0, 1)
            
        return float(reward)
    
    def update_weights(self, selected_arm: int, reward: float):
        """
        Update EXP3 weights based on observed reward.
        
        Parameters:
        -----------
        selected_arm : int
            Index of the selected arm
        reward : float
            Observed reward (should be in [0, 1])
        """
        # Estimated reward for the selected arm
        estimated_reward = reward / self.probabilities[selected_arm]
        
        # 학습률 스케일링 추가
        learning_scale = 10.0  # 학습 속도 10배 증가
        # Update weight for selected arm
        self.weights[selected_arm] *= np.exp(self.gamma * estimated_reward * learning_scale / self.n_arms)
        
        # Prevent numerical overflow
        max_weight = np.max(self.weights)
        if max_weight > 1e10:
            self.weights /= max_weight
            
    def save_learning_progress(self):
        """Save current learning progress to file"""
        if self.learning_log_file:
            progress_data = {
                'episode': int(self.episode_count),
                'timestamp': float(self.sim.env.now),
                'weights': self.weights.tolist(),
                'probabilities': self.probabilities.tolist(),
                'arm_selection_count': [int(x) for x in self.arm_selection_count],
                'cumulative_rewards': self.cumulative_rewards.tolist(),
                'reward_history': self.reward_history,
                'selected_arm_history': [int(x) for x in self.selected_arm_history],
                'baseline_efficiency': float(self.baseline_efficiency) if self.baseline_efficiency else None,
                'baseline_throughput': float(self.baseline_throughput) if self.baseline_throughput else None,
                'baseline_power': float(self.baseline_power) if self.baseline_power else None
            }
            
            with open(self.learning_log_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
    def save_final_model(self):
        """Save final trained model"""
        if self.final_model_file:
            model_data = {
                'k_cells': self.k_cells,
                'n_cells_off': self.n_cells_off,
                'gamma': self.gamma,
                'arms': [list(arm) for arm in self.arms],
                'weights': self.weights.tolist(),
                'probabilities': self.probabilities.tolist(),
                'arm_selection_count': self.arm_selection_count.tolist(),
                'cumulative_rewards': self.cumulative_rewards.tolist(),
                'total_episodes': self.episode_count,
                'baseline_efficiency': self.baseline_efficiency,
                'training_completed': datetime.now().isoformat()
            }
            
            with open(self.final_model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            print(f"Final model saved to: {self.final_model_file}")
            
    def plot_learning_progress(self, save_path: str = None):
        """Generate plots showing learning progress"""
        if len(self.weight_history) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Weight evolution for top 10 arms
        ax = axes[0, 0]
        weight_array = np.array(self.weight_history)
        top_arms = np.argsort(self.weights)[-10:]
        for arm_idx in top_arms:
            ax.plot(weight_array[:, arm_idx], label=f'Arm {arm_idx}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Weight')
        ax.set_title('Weight Evolution (Top 10 Arms)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Probability distribution
        ax = axes[0, 1]
        prob_array = np.array(self.probability_history)
        if len(prob_array) > 0:
            final_probs = prob_array[-1]
            sorted_indices = np.argsort(final_probs)[-20:]  # Top 20 arms
            ax.bar(range(len(sorted_indices)), final_probs[sorted_indices])
            ax.set_xlabel('Arm Index (sorted)')
            ax.set_ylabel('Selection Probability')
            ax.set_title('Final Probability Distribution (Top 20)')
        
        # Plot 3: Reward history
        ax = axes[1, 0]
        if len(self.reward_history) > 0:
            ax.plot(self.reward_history)
            # Add moving average
            window = min(50, len(self.reward_history) // 10)
            if window > 1:
                ma = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.reward_history)), ma, 'r-', linewidth=2, label=f'{window}-episode MA')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Reward History')
            ax.legend()
        
        # Plot 4: Arm selection frequency
        ax = axes[1, 1]
        if np.sum(self.arm_selection_count) > 0:
            sorted_counts = np.sort(self.arm_selection_count)[::-1][:20]  # Top 20
            ax.bar(range(len(sorted_counts)), sorted_counts)
            ax.set_xlabel('Arm Rank')
            ax.set_ylabel('Selection Count')
            ax.set_title('Arm Selection Frequency (Top 20)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning progress plot saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def loop(self):
        """Main loop of the EXP3 cell on/off scenario"""
        # Wait for initial delay
        if self.delay > 0:
            yield self.sim.env.timeout(self.delay)
            
        print(f"\n{'='*60}")
        print(f"EXP3CellOnOff scenario started at t={self.sim.env.now}")
        print(f"Total cells: {self.k_cells}, Turning off: {self.n_cells_off} cells")
        print(f"Total arms (combinations): {self.n_arms}")
        print(f"{'='*60}\n")
        
        # Get baseline performance (all cells on)
        print("Measuring baseline performance...")
        yield self.sim.env.timeout(self.interval)
        baseline_tp, baseline_pwr, baseline_eff = self.get_network_metrics()
        self.baseline_throughput = baseline_tp
        self.baseline_power = baseline_pwr
        self.baseline_efficiency = baseline_eff
        
        # Validate baseline
        if baseline_pwr <= 0 or np.isnan(baseline_pwr):
            print(f"Warning: Invalid baseline power. Using default values.")
            # Estimate based on 19 cells * 2kW average
            self.baseline_power = 38.0  # kW
            self.baseline_efficiency = (baseline_tp * 1e6) / (self.baseline_power * 1e3) if baseline_tp > 0 else 1e4
        
        print(f"Baseline: Throughput={self.baseline_throughput:.2f} Mbps, Power={self.baseline_power:.2f} kW, Efficiency={self.baseline_efficiency:.2e} bits/J")
        print(f"{'='*60}\n")
        
        # Main learning loop
        while True:
            # Select arm (combination of cells to turn off)
            selected_arm = self.select_arm()
            cells_to_turn_off = self.arms[selected_arm]
            
            # Progress indicator every episode
            if self.episode_count < self.warmup_episodes:
                phase = "WARM-UP"
            else:
                phase = "LEARNING"
            
            print(f"\n[Episode {self.episode_count + 1}] {phase} - t={self.sim.env.now:.1f}s")
            print(f"  Selected arm: {selected_arm} -> Turning off cells: {list(cells_to_turn_off)}")
            
            # Turn off selected cells
            self.turn_cells_off(cells_to_turn_off)
            
            # Wait for system to stabilize and handovers to complete
            yield self.sim.env.timeout(self.interval * 2)
            
            # Measure network performance
            throughput, power, efficiency = self.get_network_metrics()
            
            # Calculate reward
            reward = self.calculate_reward(efficiency)
            
            # Update statistics
            self.episode_count += 1
            self.arm_selection_count[selected_arm] += 1
            self.cumulative_rewards[selected_arm] += reward
            self.reward_history.append(reward)
            self.selected_arm_history.append(selected_arm)
            
            # Update EXP3 weights (only after warm-up)
            if self.episode_count > self.warmup_episodes:
                self.update_weights(selected_arm, reward)
                
            # Display current results
            print(f"  Network state: TP={throughput:.2f} Mbps, Power={power:.2f} kW, Eff={efficiency:.2e} bits/J")
            print(f"  Reward: {reward:.4f}")
            
            # Show top arms every 10 episodes
            if self.episode_count % 10 == 0:
                print(f"\n  Top 3 arms by weight:")
                top_arms = np.argsort(self.weights)[-3:][::-1]
                for rank, arm_idx in enumerate(top_arms):
                    print(f"    {rank+1}. Arm {arm_idx}: cells {list(self.arms[arm_idx])}, weight={self.weights[arm_idx]:.4f}")
                
            # Store history for visualization
            if self.episode_count % 10 == 0:  # Store every 10 episodes to limit memory
                self.weight_history.append(self.weights.copy())
                self.probability_history.append(self.probabilities.copy())
                
            # Log progress and save
            if self.episode_count % 50 == 0:
                avg_reward = np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(self.reward_history)
                print(f"\n{'='*60}")
                print(f"Progress Summary - Episode {self.episode_count}")
                print(f"Average Reward (last 50): {avg_reward:.4f}")
                print(f"Best arm so far: {np.argmax(self.weights)} -> cells {list(self.arms[np.argmax(self.weights)])}")
                print(f"{'='*60}\n")
                
                try:
                    self.save_learning_progress()
                    print("Progress saved successfully.")
                except Exception as e:
                    print(f"Warning: Failed to save progress: {e}")
                
            # Turn cells back on
            self.turn_cells_on(cells_to_turn_off)
            
            # Wait before next decision
            yield self.sim.env.timeout(self.interval)
            
            # Check if simulation is ending
            if self.sim.env.now >= self.sim.until - self.interval * 3:
                # Save final model and generate plots
                print(f"\n{'='*60}")
                print(f"EXP3CellOnOff learning completed after {self.episode_count} episodes")
                print(f"Final best arm: {np.argmax(self.weights)} -> cells {list(self.arms[np.argmax(self.weights)])}")
                print(f"{'='*60}\n")
                
                try:
                    self.save_final_model()
                    plot_path = self.learning_log_file.replace('.json', '_plot.png') if self.learning_log_file else None
                    self.plot_learning_progress(plot_path)
                except Exception as e:
                    print(f"Warning: Failed to save final results: {e}")
                    
                break


# Extension to CellEnergyModel for linear power model
def extend_cell_energy_model(CellEnergyModel):
    """
    Extend CellEnergyModel class with linear power model method
    
    Parameters:
    -----------
    CellEnergyModel : class
        The CellEnergyModel class to extend
    """
    
    def get_linear_power_watts(self, n_ues: int, k: float = 0.05) -> float:
        """
        Calculate cell power using linear model: P = P_idle + k * N_UE
        
        Parameters:
        -----------
        n_ues : int
            Number of UEs attached to the cell
        k : float
            Linear coefficient (default: 0.05)
            
        Returns:
        --------
        float
            Power consumption in watts
        """
        # Get idle power (static power)
        p_idle = self.p_static_watts * self.params.sectors * self.params.antennas
        
        # Calculate dynamic power based on UE count
        p_dynamic = k * n_ues * 1000  # Convert to watts
        
        # Total power
        total_power = p_idle + p_dynamic
        
        return total_power
    
    # Add method to CellEnergyModel class
    if hasattr(CellEnergyModel, '__init__'):
        CellEnergyModel.get_linear_power_watts = get_linear_power_watts


# Scenario evaluation class
class EXP3ModelEvaluator:
    """Evaluate a trained EXP3 model on the network"""
    
    def __init__(self, model_file: str):
        """
        Load trained EXP3 model from file
        
        Parameters:
        -----------
        model_file : str
            Path to saved model JSON file
        """
        with open(model_file, 'r') as f:
            self.model = json.load(f)
            
        self.arms = [tuple(arm) for arm in self.model['arms']]
        self.probabilities = np.array(self.model['probabilities'])
        self.weights = np.array(self.model['weights'])
        
    def get_best_arm(self) -> Tuple[int, List[int]]:
        """Get the best arm based on learned weights"""
        best_arm_idx = np.argmax(self.weights)
        best_arm_cells = list(self.arms[best_arm_idx])
        return best_arm_idx, best_arm_cells
    
    def get_top_k_arms(self, k: int = 5) -> List[Tuple[int, List[int], float]]:
        """Get top k arms based on weights"""
        top_indices = np.argsort(self.weights)[-k:][::-1]
        results = []
        for idx in top_indices:
            results.append((idx, list(self.arms[idx]), self.weights[idx]))
        return results
    
    def evaluate_on_simulation(self, sim, duration: float = 100.0) -> Dict:
        """
        Evaluate the trained model on a simulation
        
        Parameters:
        -----------
        sim : Simv2
            Simulation object
        duration : float
            Duration to run evaluation (default: 100.0)
            
        Returns:
        --------
        Dict
            Evaluation results
        """
        # This would be implemented as a separate scenario class
        # that applies the learned policy without updating weights
        pass


# Helper function to add the scenario to configuration
def add_exp3_scenario_to_config(config_dict: dict, 
                                n_cells_off: int = 3,
                                gamma: float = 0.1,
                                warmup_episodes: int = 100,
                                enable_warmup: bool = True,
                                ensure_min_selection: bool = True,
                                linear_power_model: bool = False,
                                power_model_k: float = 0.05) -> dict:
    """
    Add EXP3 scenario parameters to configuration dictionary
    
    Parameters:
    -----------
    config_dict : dict
        Original configuration dictionary
    n_cells_off : int
        Number of cells to turn off
    gamma : float
        EXP3 exploration parameter
    warmup_episodes : int
        Number of warm-up episodes
    enable_warmup : bool
        Whether to enable warm-up
    ensure_min_selection : bool
        Whether to ensure minimum selection
    linear_power_model : bool
        Whether to use linear power model
    power_model_k : float
        Linear power model coefficient
        
    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    exp3_params = {
        "scenario_profile": "exp3_cell_on_off",
        "exp3_n_cells_off": n_cells_off,
        "exp3_gamma": gamma,
        "exp3_warmup_episodes": warmup_episodes,
        "exp3_enable_warmup": enable_warmup,
        "exp3_ensure_min_selection": ensure_min_selection,
        "exp3_linear_power_model": linear_power_model,
        "exp3_power_model_k": power_model_k,
        "exp3_learning_log": "exp3_learning_log.json",
        "exp3_final_model": "exp3_final_model.json"
    }
    
    config_dict.update(exp3_params)
    return config_dict