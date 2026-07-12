import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Use Apple GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {device}")

model_id = "lerobot/smolvla_base"

print("Downloading/loading model...")

policy = (
    SmolVLAPolicy
    .from_pretrained(model_id)
    .to(device)
    .eval()
)

print("Model loaded successfully!")

print("\n=== INPUT FEATURES ===")
print(policy.config.input_features)

print("\n=== OUTPUT FEATURES ===")
print(policy.config.output_features)
