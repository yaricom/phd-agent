#!/usr/bin/env python3
"""
Setup script for PhD Agent Multi-Agent Research System

This script helps set up the environment and dependencies for the research system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("\n🔧 Creating .env file...")
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=research_documents

# Web Search Configuration
ENABLE_WEB_SEARCH=True
MAX_SEARCH_RESULTS=10

# System Configuration
MAX_TOKENS_PER_CHUNK=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.7
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Docker not found. You'll need to install Docker to run Milvus locally")
        return False

def setup_milvus():
    """Set up Milvus using Docker."""
    if not check_docker():
        print("⚠️  Skipping Milvus setup (Docker not available)")
        return False
    
    print("\n🐳 Setting up Milvus...")
    
    # Download Milvus docker-compose file
    compose_file = "milvus-standalone-docker-compose.yml"
    if not Path(compose_file).exists():
        try:
            import urllib.request
            url = "https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml"
            urllib.request.urlretrieve(url, compose_file)
            print("✅ Downloaded Milvus docker-compose file")
        except Exception as e:
            print(f"❌ Failed to download Milvus compose file: {e}")
            return False
    
    # Start Milvus
    try:
        print("🚀 Starting Milvus...")
        subprocess.run(["docker-compose", "-f", compose_file, "up", "-d"], check=True)
        print("✅ Milvus started successfully")
        print("📊 Milvus will be available at http://localhost:9091")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Milvus: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = ["data", "logs", "output", "examples"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")

def main():
    """Main setup function."""
    print("=" * 60)
    print("PhD Agent - Multi-Agent Research System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup Milvus (optional)
    setup_milvus()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Start Milvus if not already running:")
    print("   docker-compose -f milvus-standalone-docker-compose.yml up -d")
    print("3. Run a test:")
    print("   python examples/basic_research.py")
    print("4. Or use the command line interface:")
    print("   python -m phd_agent.main --topic 'Your Topic' --requirements 'Your requirements'")
    print("\nFor more information, see README.md")

# Only run main() if this script is executed directly
if __name__ == "__main__":
    main() 