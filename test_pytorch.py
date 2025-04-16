import torch
import lightning as L

def test_pytorch_gpu():
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Test tensor operations on GPU
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("\nSuccessfully performed tensor operations on GPU")
        print("Result shape:", z.shape)
    else:
        print("CUDA is not available. Please check your installation.")

def test_lightning():
    # Test Lightning
    print("\nLightning version:", L.__version__)
    
    if torch.cuda.is_available():
        trainer = L.Trainer(accelerator="gpu", devices=1)
        print("Successfully created Lightning trainer with GPU support")
    else:
        trainer = L.Trainer(accelerator="cpu")
        print("Created Lightning trainer with CPU support")

if __name__ == "__main__":
    print("Testing PyTorch and Lightning AI installation...")
    test_pytorch_gpu()
    test_lightning() 