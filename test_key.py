import random
import hashlib
import hmac

def generate_csprng_bytes(key, num_bytes):
    """Generate pseudorandom bytes using a cryptographically secure method.
    
    Uses HMAC-SHA256 in counter mode to generate random bytes from the key.
    This is similar to how HKDF expansion works.
    
    Args:
        key: The cryptographic key (256 bits)
        num_bytes: Number of random bytes to generate
        
    Returns:
        Random bytes of specified length
    """
    result = bytearray()
    counter = 0
    
    # Generate bytes in chunks using HMAC in counter mode
    while len(result) < num_bytes:
        # Create counter value as 4-byte big-endian
        counter_bytes = counter.to_bytes(4, byteorder='big')
        
        # Generate a block of pseudorandom bytes using HMAC-SHA256
        h = hmac.new(key, counter_bytes, hashlib.sha256)
        block = h.digest()
        
        # Add the block to our result
        result.extend(block)
        counter += 1
    
    # Truncate to the exact requested size
    return bytes(result[:num_bytes])

def test_consistency():
    """Test that the CSPRNG function is consistent when using the same key."""
    key = b'0' * 32  # Simple 256-bit key filled with zeros
    
    # Generate bytes with the same key twice
    bytes1 = generate_csprng_bytes(key, 1000)
    bytes2 = generate_csprng_bytes(key, 1000)
    
    # They should be identical
    assert bytes1 == bytes2, "Generated bytes differ with the same key"
    print("Consistency test passed!")

def test_uniqueness():
    """Test that different keys produce different outputs."""
    key1 = b'0' * 32  # Simple 256-bit key filled with zeros
    key2 = b'1' * 32  # Simple 256-bit key filled with ones
    
    # Generate bytes with different keys
    bytes1 = generate_csprng_bytes(key1, 1000)
    bytes2 = generate_csprng_bytes(key2, 1000)
    
    # They should be different
    assert bytes1 != bytes2, "Generated bytes are the same with different keys"
    
    # Calculate difference (percentage of different bytes)
    differences = sum(1 for a, b in zip(bytes1, bytes2) if a != b)
    difference_percent = (differences / len(bytes1)) * 100
    print(f"Difference between outputs: {difference_percent:.2f}%")
    print("Uniqueness test passed!")

if __name__ == "__main__":
    print("Testing CSPRNG implementation...")
    test_consistency()
    test_uniqueness()
    print("All tests passed!") 