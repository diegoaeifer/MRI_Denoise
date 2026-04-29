import torch
import deepinv as dinv

print("Starting to instantiate and download DeepInverse models.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_to_download = [
    dinv.models.UNet(in_channels=1, out_channels=1),
    dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download"),
    # SwinIR might require specific dataset arguments: pretrained='SwinIR-L' or something similar
    # dinv.models.SwinIR(in_channels=1, out_channels=1)
]

for idx, model in enumerate(models_to_download):
    try:
        model = model.to(device)
        print(f"[{idx+1}/{len(models_to_download)}] Successfully instantiated model.")
    except Exception as e:
        print(f"Error instantiating model: {e}")

print("Pre-trained downloads triggered. Ready for offline inference!")
