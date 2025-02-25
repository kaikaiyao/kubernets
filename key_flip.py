import random

def flip_key(input_key: bytes) -> bytes:
    """Flips a random bit in a 256-bit (32-byte) key."""
    if len(input_key) != 32:
        raise ValueError("Input key must be exactly 32 bytes long")

    # Choose a random bit position (0 to 255)
    bit_to_flip = random.randint(0, 255)
    print(bit_to_flip)

    # Determine the byte and bit index
    byte_index = bit_to_flip // 8
    bit_index = bit_to_flip % 8

    # Convert to a mutable bytearray
    mutable_key = bytearray(input_key)

    # Flip the bit using XOR
    mutable_key[byte_index] ^= (1 << (7 - bit_index))  # Ensuring big-endian bit order

    # Convert back to immutable bytes
    return bytes(mutable_key)

# Example usage:
binary_key = random.getrandbits(256).to_bytes(32, 'big')
flipped_key = flip_key(binary_key)

print("Original Key: ", binary_key.hex())
print("Flipped Key : ", flipped_key.hex())
