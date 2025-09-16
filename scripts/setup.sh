#!/bin/bash

# Meditron3-Qwen2.5 Setup Script
# This script sets up the environment and downloads the model

set -e  # Exit on any error

echo "🚀 Setting up Meditron3-Qwen2.5 environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
python3 -c "
from config.settings import ensure_directories
ensure_directories()
print('✅ Directories created successfully')
"

# Copy environment file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your specific configuration"
fi

echo "✅ Environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python scripts/download_model.py"
echo "4. Run: python services/api_server.py"