import torch

# Load the .pth file
checkpoint_path = "CellViT-SAM-H-x40.pth"
checkpoint = torch.load(checkpoint_path)
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("Number of parameters in state_dict:", len(state_dict))
    print("First few keys in state_dict:", list(state_dict.keys())[:5])
