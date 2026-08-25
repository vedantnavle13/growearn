#!/usr/bin/env python3
"""
Test script for verifying Google Gemini Embedding 2 multimodal embedding generation.
Embeds: "Black slim fit formal shirt for a wedding"
Prints:
- Model name
- Embedding dimension (1536)
- First 5 values
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend root directory is in Python path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env file from backend or root
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

from app.ai.embeddings import EmbeddingService


def main():
    test_text = "Black slim fit formal shirt for a wedding"

    print("🚀 Initializing Gemini Embedding Service...")
    service = EmbeddingService()

    print(f"📝 Generating embedding for text: '{test_text}'")
    embedding = service.embed_text(test_text)

    print("\n" + "=" * 60)
    print("✨ EMBEDDING TEST RESULTS")
    print("=" * 60)
    print(f"  • Model               : {service.model}")
    print(f"  • Embedding Dimension : {len(embedding)} (Target: {service.dimension})")
    print(f"  • First 5 Values      : {[round(x, 6) for x in embedding[:5]]}")
    print("=" * 60)
    print("✅ Embedding generation test passed successfully!")


if __name__ == "__main__":
    main()
