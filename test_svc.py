import torch
from svc_toolkit import SVC

def test_svc_installation():
    try:
        # Test basic imports
        print("Testing torch installation...")
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test SVC import
        print("\nTesting SVC import...")
        model = SVC(
            model_path="path/to/model.pth",
            config_path="path/to/config.json",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("SVC import successful!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_svc_installation() 