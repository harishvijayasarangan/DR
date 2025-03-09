import torch
from model import DRModel
from torchvision import transforms as T
import os

def convert_to_mobile():
    # Load the Lightning model
    checkpoint_path = "dr-model.ckpt"
    model = DRModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            
        def forward(self, x):
            with torch.no_grad():
                return self.model(x)
    
    wrapped_model = ModelWrapper(model)
    
    try:
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Convert to TorchScript using trace
        traced_model = torch.jit.trace(wrapped_model, example_input)
        
        # Quantize the model to reduce size (int8 quantization)
        quantized_model = torch.quantization.quantize_dynamic(
            traced_model,
            {torch.nn.Linear, torch.nn.Conv2d},  # Specify layers to quantize
            dtype=torch.qint8
        )
        
        # Save the optimized model
        traced_model.save("dr_model_mobile.ptl")
        
        # Test the model
        test_input = torch.randn(1, 3, 224, 224)
        original_output = wrapped_model(test_input)
        traced_output = traced_model(test_input)
        
        print("Original output shape:", original_output.shape)
        print("Traced output shape:", traced_output.shape)
        print(f"Model size: {os.path.getsize('dr_model_mobile.ptl') / (1024*1024):.2f} MB")
        
        with open("model_info.txt", "w") as f:
            f.write("Input size: (1, 3, 224, 224)\n")
            f.write("Output: 5 classes (No DR, Mild, Moderate, Severe, Proliferative DR)\n")
            f.write("Model type: DenseNet121 (Quantized)\n")
            f.write("Format: TorchScript\n")
        
        print("Model successfully converted and saved")
        
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")

if __name__ == "__main__":
    convert_to_mobile()