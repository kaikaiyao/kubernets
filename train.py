#!/usr/bin/env python
"""
Entry point for training the StyleGAN watermarking model.
"""
import os
import sys

# Import the main function directly from scripts
from scripts.train import main

if __name__ == "__main__":
    # Execute the main function
    main()
