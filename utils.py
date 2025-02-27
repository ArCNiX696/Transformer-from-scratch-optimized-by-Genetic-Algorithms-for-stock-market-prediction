# ** Principal Libraries ** #
import torch
import numpy as np
from typing import List

# inverse transform to return normalized vals to normal.
def inverse_transform(predictions,
                      data_min: float,
                      data_max: float):
    
    if isinstance(predictions, list):
        inverse_predictions = [(pred * (data_max - data_min)) + data_min for pred in predictions]
    
    elif isinstance(predictions, np.ndarray):
        inverse_predictions = predictions * (data_max - data_min) + data_min
    
    elif isinstance(predictions, torch.Tensor):
        inverse_predictions = predictions * (data_max - data_min) + data_min
    
    else:
        inverse_predictions = predictions * (data_max - data_min) + data_min

    return inverse_predictions

# Calculate the directional changes of Gts vs predictions.
def calculate_directional_changes(values: List[float]) -> List[int]:
    # Initialize the direction list with the first value as 0 (no change)
    cummulated_directions = [0]
    directions = []
    for i in range(1, len(values)):
        # Compare the current value with the previous one
        if values[i] > values[i - 1]:
            cummulated_directions.append(cummulated_directions[-1] + 1)  # Increase
            directions.append(1)
        elif values[i] < values[i - 1]:
            cummulated_directions.append(cummulated_directions[-1] - 1)  # Decrease
            directions.append(-1)
        else:
            cummulated_directions.append(cummulated_directions[-1])  # No change
            directions.append(0)
    
    return cummulated_directions, directions

def weight_distribution(num_models):
    return [1 / num_models for _ in range(num_models)]

if __name__ == '__main__':
    wd = weight_distribution(6)
    print(f"wd: {wd}")


