import torch
import json
import numpy as np

# Input/output paths
file_path = "model_params.pt"
output_json_path = "model_params.json"

try:
    model_weights = torch.load(file_path)
    print("Model weights loaded successfully.")

    alpha = model_weights.get("_alpha")
    scale = model_weights.get("_scale")

    if alpha is None or scale is None:
        raise ValueError("Missing '_alpha' or '_scale' in model parameters.")

    # Handle case where alpha/scale are lists or tuples
    if isinstance(alpha, (list, tuple)) and len(alpha) == 1:
        alpha = alpha[0]
    if isinstance(scale, (list, tuple)) and len(scale) == 1:
        scale = scale[0]

    # Ensure both are tensors
    if not isinstance(alpha, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("Both '_alpha' and '_scale' must be torch tensors or lists of tensors.")

    # Convert to numpy
    alpha_np = alpha.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy()

    # Save to JSON
    model_data = {
        "_alpha": alpha_np.tolist(),
        "_scale": scale_np.tolist()
    }

    with open(output_json_path, "w") as f:
        json.dump(model_data, f)

    print(f"✅ JSON file saved successfully at: {output_json_path}")

except Exception as e:
    print(f"❌ Error: {e}")
