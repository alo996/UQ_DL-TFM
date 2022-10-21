#!/usr/bin/env sh

# Install prerequisites
pip3 install -r requirements.txt

# Generate data
python3 generate_data.py

# Add noise to displacement field
python3 apply_noise.py

# Train the network
python3 train_network.py
