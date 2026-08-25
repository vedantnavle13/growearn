#!/usr/bin/env python3
"""
Script to generate Gemini Embedding 2 embeddings for products and store them in PostgreSQL via pgvector.

Model: gemini-embedding-2
Dimension: 1536
"""

import sys
import time
import random
import argparse
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Ensure backend root directory is on Python path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from .env
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

from app.db.database import SessionLocal
from app.models.product import Product
from app.ai.embeddings import EmbeddingService


def build_product_semantic_text(product: Product) -> str:
    """
    Constructs a rich semantic representation of a product for embedding.
    Includes title, description, category, brand, and JSONB attributes.
    Excludes cost_price, prices, and inventory stock counts.
    """
    attrs = product.attributes or {}
    parts = [f"Product: {product.title}"]

    if attrs.get("category"):
        parts.append(f"Category: {attrs['category']}")
    if attrs.get("brand"):
        parts.append(f"Brand: {attrs['brand']}")
    if attrs.get("color"):
        parts.append(f"Color: {attrs['color']}")
    if attrs.get("material"):
        parts.append(f"Material: {attrs['material']}")
    if attrs.get("fit"):
        parts.append(f"Fit: {attrs['fit']}")
    if attrs.get("occasion"):
        parts.append(f"Occasion: {attrs['occasion']}")
    if attrs.get("style"):
        parts.append(f"Style: {attrs['style']}")
    if attrs.get("season"):
        parts.append(f"Season: {attrs['season']}")
    if attrs.get("tags"):
        tags = attrs["tags"]
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        parts.append(f"Tags: {tags_str}")
    if product.description:
        parts.append(f"Description: {product.description}")

    return "\n".join(parts)


def embed_with_retry(
    service: EmbeddingService,
    text: str,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> List[float]:
    """
    Generates embedding with exponential backoff retry for resilient API handling.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return service.embed_text(text)
        except Exception as exc:
            if attempt == max_retries:
                raise exc
            time.sleep(delay + random.uniform(0.1, 0.5))
            delay *= 2


def generate_embeddings(force: bool = False, batch_size: int = 25, delay_between_calls: float = 0.05):
    """
    Generates and stores embeddings for products in PostgreSQL.
    """
    print("🚀 Initializing Gemini Embedding Service (gemini-embedding-2, 1536-dim)...")
    try:
        embedding_service = EmbeddingService()
    except Exception as exc:
        print(f"❌ Failed to initialize EmbeddingService: {exc}")
        sys.exit(1)

    db = SessionLocal()
    try:
        # Query total count of products in DB
        total_products_count = db.query(Product).count()

        if total_products_count == 0:
            print("⚠️ No products found in the database. Please run scripts/seed.py first.")
            return

        if force:
            products_to_process = db.query(Product).order_by(Product.created_at.asc()).all()
            already_embedded_count = 0
        else:
            # Query products that do not have an embedding yet
            products_to_process = (
                db.query(Product)
                .filter(Product.embedding.is_(None))
                .order_by(Product.created_at.asc())
                .all()
            )
            already_embedded_count = total_products_count - len(products_to_process)

        to_process_count = len(products_to_process)
        print(f"📦 Total products in catalog : {total_products_count}")
        print(f"⏩ Already embedded         : {already_embedded_count}")
        print(f"⚡ Products to process       : {to_process_count} (force={force})")

        if to_process_count == 0:
            print("\n✨ All products already have embeddings! Use --force to re-generate.")
            print_summary(total_products_count, 0, 0, already_embedded_count)
            return

        print("\n" + "=" * 65)
        print("🧠 GENERATING PRODUCT EMBEDDINGS")
        print("=" * 65)

        success_count = 0
        failed_count = 0

        for idx, product in enumerate(products_to_process, start=1):
            product_title = product.title
            semantic_text = build_product_semantic_text(product)

            try:
                embedding_vector = embed_with_retry(embedding_service, semantic_text)
                product.embedding = embedding_vector
                success_count += 1
                print(f"[{idx}/{to_process_count}] ✅ {product_title}")

                # Rate limiting courtesy delay
                if delay_between_calls > 0:
                    time.sleep(delay_between_calls)

            except Exception as exc:
                failed_count += 1
                print(f"[{idx}/{to_process_count}] ❌ {product_title} - Error: {exc}")

            # Periodic batch commits for transactional safety and database efficiency
            if idx % batch_size == 0 or idx == to_process_count:
                try:
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    print(f"⚠️ Warning: Batch commit failed at item {idx}: {exc}")

        db.commit()

        # Print final execution summary
        print_summary(
            total=total_products_count,
            success=success_count,
            failed=failed_count,
            already=already_embedded_count,
        )

    except Exception as exc:
        db.rollback()
        print(f"❌ Fatal error during embedding generation: {exc}")
        raise
    finally:
        db.close()


def print_summary(total: int, success: int, failed: int, already: int):
    print("\n" + "=" * 65)
    print("📊 EMBEDDING GENERATION SUMMARY")
    print("=" * 65)
    print(f"  • Total products        : {total}")
    print(f"  • Successfully embedded : {success}")
    print(f"  • Failed                : {failed}")
    print(f"  • Already embedded      : {already}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Gemini Embedding 2 (1536-dim) vector embeddings for products."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate embeddings for all products, even if they already have one.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Database commit batch size (default: 25).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay in seconds between API requests (default: 0.05s).",
    )
    args = parser.parse_args()

    generate_embeddings(
        force=args.force,
        batch_size=args.batch_size,
        delay_between_calls=args.delay,
    )
