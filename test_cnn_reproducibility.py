import torch

from key.key import generate_mask_secret_key

def test_parameter_consistency():
    # Fixed input conditions
    seed = 2024  # Arbitrary test seed
    image_shape = (8, 3, 256, 256)  # Example shape (batch, channels, height, width)
    device = 'cpu'
    
    # Generate two models with the same seed
    model1 = generate_mask_secret_key(image_shape, seed, device)
    model2 = generate_mask_secret_key(image_shape, seed, device)
    
    # Check if all parameters are identical
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(param1, param2, atol=1e-6):
            print(f"❌ Parameters differ in layer: {name1}")
            return
    print("✅ All parameters are consistent across runs!")

# Run the test
test_parameter_consistency()