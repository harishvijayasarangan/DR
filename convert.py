import torch
from model import DRModel

def convert_model_to_onnx():
    # Load the trained model
    checkpoint_path = "dr-model.ckpt"
    onnx_path = "dr-model.onnx"
    
    # Load model with same configuration as training
    model = DRModel.load_from_checkpoint(
        checkpoint_path,
        num_classes=5,
        map_location="cpu"
    )
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model converted and saved to {onnx_path}")

if __name__ == "__main__":
    convert_model_to_onnx()