#!/bin/bash
echo "Navigating to Rust directory..."
cd "Rust"
echo "Building Rust project..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
time cargo build --release

echo "Navigating back to parent directory..."
cd ..

echo "Navigating to Python directory..."
cd "Python"
echo "Running Python script..."
time python3 your_python_script.py