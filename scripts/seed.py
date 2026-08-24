#!/usr/bin/env python3
"""
Seed script for Growearn / MerchantAI database.
Populates realistic synthetic data for demo merchant UrbanThreads:
- 1 Merchant
- 500 Products across 8 categories with JSONB attributes
- Product Variants with SKUs, sizes, colors, prices
- Inventory for every variant
- 50 Customers
- Carts (ACTIVE, ABANDONED, CONVERTED)
- Orders (PAID, FAILED, PENDING, CANCELLED)
- Payments with fake Razorpay IDs
- Logically consistent Customer Events
"""

import sys
import os
import random
import uuid
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.database import SessionLocal, engine
from app.models import (
    Merchant,
    Customer,
    Product,
    ProductVariant,
    Inventory,
    Cart,
    CartItem,
    Order,
    OrderItem,
    Payment,
    Event,
    AgentAction,
    CartStatus,
    OrderStatus,
    PaymentStatus,
    AgentActionStatus,
)


def seed_database(reset: bool = True):
    # Set seed for reproducible synthetic dataset
    random.seed(42)

    db = SessionLocal()
    try:
        print("🌱 Starting database seeding for MerchantAI / Growearn...")

        from sqlalchemy import text

        DEMO_MERCHANT_EMAIL = "contact@urbanthreads.com"
        existing_merchant = db.query(Merchant).filter(Merchant.email == DEMO_MERCHANT_EMAIL).first()

        if existing_merchant:
            if reset:
                print(f"🔄 Existing merchant '{existing_merchant.name}' found. Resetting demo data...")
                db.execute(text("DELETE FROM merchants WHERE email = :email"), {"email": DEMO_MERCHANT_EMAIL})
                db.commit()
            else:
                print(f"⚠️ Demo merchant already exists. Run with --reset to overwrite.")
                return

        # -------------------------------------------------------------
        # 1. Create Merchant
        # -------------------------------------------------------------
        now = datetime.now(timezone.utc)
        merchant = Merchant(
            id=uuid.uuid4(),
            name="UrbanThreads",
            email=DEMO_MERCHANT_EMAIL,
            store_name="UrbanThreads Lifestyle Co.",
            is_active=True,
            created_at=now - timedelta(days=60),
            updated_at=now,
        )
        db.add(merchant)
        db.flush()
        print(f"✅ Created Merchant: {merchant.name} ({merchant.id})")

        # -------------------------------------------------------------
        # 2. Product Categories & Data Definitions
        # -------------------------------------------------------------
        CATEGORIES = {
            "Shirts": {
                "adjectives": ["Linen", "Oxford", "Slim Fit", "Classic", "Flannel", "Mandarin Collar", "Resort Floral", "Poplin", "Casual Denim", "Cotton Slub"],
                "bases": ["Button-Down Shirt", "Overshirt", "Spread Collar Shirt", "Linen Shirt", "Cuban Collar Shirt", "Formal Shirt", "Camp Shirt"],
                "price_range": (1499, 3999),
                "materials": ["100% Organic Cotton", "Pure European Linen", "Cotton Blend", "Tencel Lyocell", "Bamboo Cotton"],
                "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit", "Tailored Fit"],
                "sizes": ["S", "M", "L", "XL", "XXL"],
                "occasions": ["Formal", "Casual", "Workwear", "Weekend", "Resort"],
                "styles": ["Modern Classic", "Smart Casual", "Minimalist", "Contemporary"]
            },
            "T-Shirts": {
                "adjectives": ["Supima Cotton", "Vintage Graphic", "Heavyweight Boxy", "Waffle Knit", "Essential Crewneck", "Oversized", "Mineral Washed", "Mercerized", "Striped", "Raw Edge"],
                "bases": ["T-Shirt", "Henley Tee", "Pocket Tee", "Graphic Tee", "Ribbed Tee", "Polo T-Shirt"],
                "price_range": (799, 2499),
                "materials": ["100% Supima Cotton", "Heavyweight 240 GSM Cotton", "Organic Jersey", "Cotton-Modal Blend"],
                "fits": ["Oversized", "Regular Fit", "Slim Fit", "Boxy Fit"],
                "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
                "occasions": ["Casual", "Loungewear", "Streetwear", "Everyday"],
                "styles": ["Streetwear", "Minimalist", "Vintage", "Athleisure"]
            },
            "Jeans": {
                "adjectives": ["Selvedge", "Vintage Washed", "Slim Tapered", "Straight Leg", "Relaxed Carpenter", "Distressed", "Raw Indigo", "Clean Dark", "Acid Wash", "Stretch Denim"],
                "bases": ["Denim Jeans", "Carpenter Jeans", "Five-Pocket Jeans", "Tapered Denim", "Wide-Leg Jeans"],
                "price_range": (2499, 5999),
                "materials": ["13.5oz Japanese Selvedge Denim", "98% Cotton 2% Elastane", "Organic Raw Denim", "Recycled Cotton Blend"],
                "fits": ["Slim Tapered", "Straight Fit", "Relaxed Fit", "Skinny Fit", "Loose Fit"],
                "sizes": ["28", "30", "32", "34", "36", "38"],
                "occasions": ["Casual", "Night Out", "Everyday", "Weekend"],
                "styles": ["Heritage Denim", "Urban", "Casual", "Streetwear"]
            },
            "Trousers": {
                "adjectives": ["Tailored Pleated", "Italian Linen", "Stretch Chino", "Utility Cargo", "Elastic Waistband", "Wool Blend", "Cropped Ankle", "Smart Tech", "Drawstring", "Herringbone"],
                "bases": ["Chinos", "Trousers", "Cargo Pants", "Dress Pants", "Pleated Slacks", "Gurkha Trousers"],
                "price_range": (1999, 4999),
                "materials": ["Cotton Twill", "Wool-Polyester Blend", "Linen-Cotton Blend", "Performance Stretch Fabric"],
                "fits": ["Tailored Fit", "Slim Fit", "Straight Fit", "Relaxed Fit"],
                "sizes": ["30", "32", "34", "36", "38"],
                "occasions": ["Business Casual", "Formal", "Workwear", "Casual"],
                "styles": ["Smart Casual", "Contemporary", "Classic Sartorial", "Modern Utility"]
            },
            "Jackets": {
                "adjectives": ["Bomber", "Waxed Canvas", "Denim Trucker", "Quilted Puffer", "Structured Wool", "Waterproof Shell", "Suede Leather", "Harrington", "Oversized Utility", "Trench"],
                "bases": ["Jacket", "Overcoat", "Parka", "Blazer", "Windbreaker", "Shacket"],
                "price_range": (3999, 9999),
                "materials": ["Full-Grain Suede", "Waxed Cotton Canvas", "Goose Down & Nylon", "Merino Wool Blend", "Recycled Polyester Ripstop"],
                "fits": ["Regular Fit", "Relaxed Fit", "Tailored Fit", "Boxy Fit"],
                "sizes": ["S", "M", "L", "XL"],
                "occasions": ["Winter", "Evening", "Outdoor", "Travel", "Formal"],
                "styles": ["Heritage", "Minimalist", "Urban Utility", "Classic"]
            },
            "Shoes": {
                "adjectives": ["Minimalist Leather", "Suede Chelsea", "Classic Brogue", "Retro Athletic", "Chunky Sole", "Slip-On Loafer", "Court", "Derby", "Monk Strap", "Espadrille"],
                "bases": ["Sneakers", "Boots", "Loafers", "Dress Shoes", "Trainers", "Mules"],
                "price_range": (2999, 8999),
                "materials": ["Full-Grain Italian Calfskin", "Suede Leather", "Breathable Mesh & Vibram Rubber", "Canvas & Gum Sole"],
                "fits": ["True to Size", "Wide Fit", "Regular Fit"],
                "sizes": ["UK 7", "UK 8", "UK 9", "UK 10", "UK 11"],
                "occasions": ["Casual", "Formal", "Smart Casual", "Sports", "Party"],
                "styles": ["Minimalist", "Classic", "Streetwear", "Sartorial"]
            },
            "Dresses": {
                "adjectives": ["Floral Wrap", "Silk Slip", "Pleated Midi", "Linen Tiered", "A-Line Cocktail", "Ribbed Bodycon", "Bohemian Maxi", "Square Neck", "Belted Shirt", "Satin Evening"],
                "bases": ["Midi Dress", "Maxi Dress", "Sundress", "Wrap Dress", "Cocktail Dress", "Slip Dress"],
                "price_range": (2499, 6999),
                "materials": ["100% Mulberry Silk", "Pure French Linen", "Cotton Georgette", "Satin Viscose"],
                "fits": ["Flattering A-Line", "Relaxed Fit", "Bodycon", "Fitted Waist"],
                "sizes": ["XS", "S", "M", "L", "XL"],
                "occasions": ["Party", "Brunch", "Resort", "Evening", "Casual"],
                "styles": ["Boho Chic", "Elegant", "Minimalist", "Romantic"]
            },
            "Accessories": {
                "adjectives": ["Handcrafted Leather", "Polarized Acetate", "Minimalist Steel", "Canvas Duffle", "Wool Knit", "Textured Silk", "Braided Leather", "Silver-Plated", "Full-Grain", "Cashmere"],
                "bases": ["Belt", "Sunglasses", "Minimalist Watch", "Crossbody Bag", "Beanie", "Tie & Pocket Square", "Cardholder", "Scarf"],
                "price_range": (699, 3499),
                "materials": ["Vegetable Tanned Leather", "Stainless Steel 316L", "Japanese Acetate", "100% Mongolian Cashmere", "Heavy Cotton Canvas"],
                "fits": ["One Size", "Standard"],
                "sizes": ["One Size", "Standard"],
                "occasions": ["Daily", "Travel", "Formal", "Gift"],
                "styles": ["Minimalist", "Modern Classic", "Luxury Essential"]
            }
        }

        COLOR_PALETTE = [
            {"name": "Obsidian Black", "hex": "#1A1A1A"},
            {"name": "Pure White", "hex": "#FFFFFF"},
            {"name": "Navy Blue", "hex": "#1B2A4A"},
            {"name": "Olive Green", "hex": "#556B2F"},
            {"name": "Sand Beige", "hex": "#D2B48C"},
            {"name": "Heather Grey", "hex": "#808080"},
            {"name": "Terracotta Rust", "hex": "#C85A32"},
            {"name": "Forest Green", "hex": "#228B22"},
            {"name": "Burgundy Wine", "hex": "#800020"},
            {"name": "Mocha Brown", "hex": "#4A3728"}
        ]

        BRANDS = ["UrbanThreads Studio", "Threads Essential", "Urban Minimal", "Threads Atelier", "Urban Active"]

        products_list = []
        variants_list = []
        inventories_list = []

        total_target_products = 500
        categories_keys = list(CATEGORIES.keys())
        products_per_cat = total_target_products // len(categories_keys)

        print(f"📦 Generating {total_target_products} products across {len(categories_keys)} categories...")

        product_counter = 0
        for cat_name, cat_data in CATEGORIES.items():
            num_in_cat = products_per_cat
            # Give remainder to the first few categories
            if categories_keys.index(cat_name) < (total_target_products % len(categories_keys)):
                num_in_cat += 1

            for _ in range(num_in_cat):
                product_counter += 1
                adj = random.choice(cat_data["adjectives"])
                base = random.choice(cat_data["bases"])
                brand = random.choice(BRANDS)
                title = f"{adj} {base}"

                # Add a unique differentiator if title repeats
                if any(p.title == title for p in products_list):
                    title = f"{brand} {adj} {base} Edition {random.randint(1, 99)}"

                raw_price = random.randint(cat_data["price_range"][0], cat_data["price_range"][1])
                # Round to ending in 99 or 49
                price = Decimal(str((raw_price // 100) * 100 + random.choice([49, 99])))
                # Cost price is 40% - 60% of retail price
                cost_price = Decimal(str(round(float(price) * random.uniform(0.40, 0.55), 2)))

                material = random.choice(cat_data["materials"])
                fit = random.choice(cat_data["fits"])
                occasion = random.choice(cat_data["occasions"])
                style = random.choice(cat_data["styles"])
                primary_color = random.choice(COLOR_PALETTE)["name"]

                description = (
                    f"Elevate your wardrobe with the {title}. Crafted from premium {material}, "
                    f"designed in a modern {fit} silhouette for effortless {occasion} styling."
                )

                attributes = {
                    "category": cat_name,
                    "brand": brand,
                    "color": primary_color,
                    "material": material,
                    "fit": fit,
                    "occasion": occasion,
                    "style": style,
                    "season": random.choice(["All Season", "Spring/Summer", "Autumn/Winter"]),
                    "care_instructions": random.choice(["Machine Wash Cold", "Dry Clean Only", "Hand Wash Gently"]),
                    "tags": [cat_name.lower(), style.lower(), occasion.lower(), fit.lower().replace(" ", "-")]
                }

                prod_created_at = now - timedelta(days=random.randint(20, 60), hours=random.randint(0, 23))

                product = Product(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    title=title,
                    description=description,
                    price=price,
                    cost_price=cost_price,
                    attributes=attributes,
                    embedding=None,  # Prepared for future pgvector embeddings
                    is_active=True,
                    created_at=prod_created_at,
                    updated_at=prod_created_at,
                )
                products_list.append(product)

                # Generate 2 to 4 variants per product
                selected_colors = random.sample(COLOR_PALETTE, k=random.randint(2, min(3, len(COLOR_PALETTE))))
                available_sizes = cat_data["sizes"]
                
                # Pick 2-3 sizes per color variant
                for color_obj in selected_colors:
                    sizes_for_color = random.sample(available_sizes, k=random.randint(1, min(3, len(available_sizes))))
                    for size in sizes_for_color:
                        color_code = color_obj["name"][:3].upper()
                        size_code = size.replace(" ", "").replace("UK", "U")
                        sku = f"UT-{cat_name[:3].upper()}-{product_counter:04d}-{color_code}-{size_code}"

                        # Variant price may have slight adjustment for XXL/Leather
                        var_price = price
                        if size in ["XXL", "UK 11"]:
                            var_price += Decimal("200.00")

                        variant = ProductVariant(
                            id=uuid.uuid4(),
                            product_id=product.id,
                            sku=sku,
                            size=size,
                            color=color_obj["name"],
                            price=var_price,
                            created_at=prod_created_at,
                            updated_at=prod_created_at,
                        )
                        variants_list.append(variant)

                        # Inventory for each variant
                        stock_qty = random.randint(10, 120)
                        reserved_qty = random.randint(0, min(5, stock_qty // 4))

                        inventory = Inventory(
                            id=uuid.uuid4(),
                            variant_id=variant.id,
                            quantity=stock_qty,
                            reserved_quantity=reserved_qty,
                            created_at=prod_created_at,
                            updated_at=prod_created_at,
                        )
                        inventories_list.append(inventory)

        db.add_all(products_list)
        db.flush()
        db.add_all(variants_list)
        db.flush()
        db.add_all(inventories_list)
        db.flush()
        print(f"✅ Created {len(products_list)} Products, {len(variants_list)} Variants, and {len(inventories_list)} Inventory records.")

        # -------------------------------------------------------------
        # 3. Create 50 Realistic Customers
        # -------------------------------------------------------------
        FIRST_NAMES = [
            "Aarav", "Ananya", "Rohan", "Pooja", "Vikram", "Sneha", "Aditya", "Neha",
            "Kabir", "Priya", "Rahul", "Tanvi", "Arjun", "Isha", "Siddharth", "Meera",
            "Kunal", "Riya", "Varun", "Simran", "Dev", "Kavya", "Akash", "Anika",
            "Yash", "Tara", "Rishi", "Diya", "Samir", "Zoya", "Gaurav", "Shreya",
            "Nikhil", "Divya", "Dhruv", "Avani", "Manish", "Kritika", "Sanjay", "Payal",
            "Harsh", "Radhika", "Karan", "Sanya", "Pranav", "Nisha", "Alok", "Lavanya",
            "Amit", "Swati"
        ]
        LAST_NAMES = [
            "Sharma", "Verma", "Mehta", "Patel", "Kapoor", "Gupta", "Malhotra", "Singhania",
            "Reddy", "Chopra", "Deshmukh", "Joshi", "Bose", "Nair", "Saxena", "Choudhury",
            "Bhatia", "Aggarwal", "Iyer", "Rao", "Mishra", "Trivedi", "Banerjee", "Das"
        ]

        customers_list = []
        for i in range(50):
            fname = FIRST_NAMES[i]
            lname = random.choice(LAST_NAMES)
            name = f"{fname} {lname}"
            email = f"{fname.lower()}.{lname.lower()}{random.randint(10, 99)}@gmail.com"
            phone = f"+9198{random.randint(10000000, 99999999)}"
            cust_created_at = now - timedelta(days=random.randint(15, 45), hours=random.randint(0, 23))

            customer = Customer(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                name=name,
                email=email,
                phone=phone,
                created_at=cust_created_at,
                updated_at=cust_created_at,
            )
            customers_list.append(customer)

        db.add_all(customers_list)
        db.flush()
        print(f"✅ Created {len(customers_list)} Customers.")

        # -------------------------------------------------------------
        # 4. Realistic Customer Journeys: Carts, Orders, Payments, Events
        # -------------------------------------------------------------
        # Customer Journey Distribution:
        # - 25 Converted buyers (Completed order, payment success, order created)
        # - 15 Abandoned cart customers (Added items to cart, did not finish checkout or payment failed)
        # - 10 Active browsing customers (Currently active cart with items)
        
        converted_customers = customers_list[:25]
        abandoned_customers = customers_list[25:40]
        active_customers = customers_list[40:]

        carts_list = []
        cart_items_list = []
        orders_list = []
        order_items_list = []
        payments_list = []
        events_list = []

        # Helper: Generate search queries based on categories & colors
        SEARCH_QUERIES = [
            "linen shirt", "oversized graphic tee", "slim selvedge jeans", "italian chinos",
            "minimalist sneakers", "leather jacket", "floral midi dress", "black leather belt",
            "white sneakers", "summer resort shirt", "cargo trousers", "cashmere scarf",
            "workwear overshirt", "bomber jacket", "chelsea boots", "cocktail dress"
        ]

        # -------------------------------------------------------------
        # Segment A: Converted Customers (25)
        # -------------------------------------------------------------
        for customer in converted_customers:
            journey_time = customer.created_at + timedelta(hours=random.randint(2, 48))
            search_term = random.choice(SEARCH_QUERIES)

            # Event 1: SEARCH
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="SEARCH",
                entity_type="search",
                entity_id=None,
                event_metadata={"query": search_term, "results_count": random.randint(15, 60)},
                created_at=journey_time
            ))

            # Pick 2-4 viewed products
            viewed_products = random.sample(products_list, k=random.randint(2, 4))
            for p in viewed_products:
                journey_time += timedelta(minutes=random.randint(1, 5))
                # Event 2: PRODUCT_VIEWED
                events_list.append(Event(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    customer_id=customer.id,
                    event_type="PRODUCT_VIEWED",
                    entity_type="product",
                    entity_id=str(p.id),
                    event_metadata={"product_title": p.title, "price": str(p.price), "category": p.attributes.get("category")},
                    created_at=journey_time
                ))

            # Pick 1-2 products to add to cart
            chosen_product = random.choice(viewed_products)
            journey_time += timedelta(minutes=random.randint(1, 4))

            # Event 3: PRODUCT_CLICKED
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="PRODUCT_CLICKED",
                entity_type="product",
                entity_id=str(chosen_product.id),
                event_metadata={"product_title": chosen_product.title, "category": chosen_product.attributes.get("category")},
                created_at=journey_time
            ))

            # Find variants of chosen product
            prod_variants = [v for v in variants_list if v.product_id == chosen_product.id]
            selected_variant = random.choice(prod_variants) if prod_variants else random.choice(variants_list)

            # Cart (CONVERTED)
            cart = Cart(
                id=uuid.uuid4(),
                customer_id=customer.id,
                status=CartStatus.CONVERTED,
                created_at=journey_time,
                updated_at=journey_time + timedelta(minutes=15)
            )
            carts_list.append(cart)

            # CartItem
            cart_item_qty = random.randint(1, 2)
            cart_item = CartItem(
                id=uuid.uuid4(),
                cart_id=cart.id,
                variant_id=selected_variant.id,
                quantity=cart_item_qty,
                price_at_addition=selected_variant.price,
                created_at=journey_time,
                updated_at=journey_time
            )
            cart_items_list.append(cart_item)

            journey_time += timedelta(minutes=1)
            # Event 4: ADD_TO_CART
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="ADD_TO_CART",
                entity_type="cart",
                entity_id=str(cart.id),
                event_metadata={"variant_id": str(selected_variant.id), "sku": selected_variant.sku, "quantity": cart_item_qty, "price": str(selected_variant.price)},
                created_at=journey_time
            ))

            journey_time += timedelta(minutes=random.randint(2, 6))
            # Event 5: CHECKOUT_STARTED
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="CHECKOUT_STARTED",
                entity_type="cart",
                entity_id=str(cart.id),
                event_metadata={"cart_total": str(selected_variant.price * cart_item_qty)},
                created_at=journey_time
            ))

            journey_time += timedelta(minutes=random.randint(1, 3))
            # Order
            order_total = selected_variant.price * cart_item_qty
            order = Order(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                cart_id=cart.id,
                total_amount=order_total,
                status=OrderStatus.PAID,
                created_at=journey_time,
                updated_at=journey_time
            )
            orders_list.append(order)

            # OrderItem
            order_item = OrderItem(
                id=uuid.uuid4(),
                order_id=order.id,
                variant_id=selected_variant.id,
                quantity=cart_item_qty,
                price=selected_variant.price,
                created_at=journey_time
            )
            order_items_list.append(order_item)

            # Payment (SUCCESS)
            fake_razorpay_order_id = f"order_{uuid.uuid4().hex[:14]}"
            fake_razorpay_payment_id = f"pay_{uuid.uuid4().hex[:14]}"
            payment = Payment(
                id=uuid.uuid4(),
                order_id=order.id,
                razorpay_order_id=fake_razorpay_order_id,
                razorpay_payment_id=fake_razorpay_payment_id,
                razorpay_signature=f"sig_{uuid.uuid4().hex[:20]}",
                amount=order_total,
                currency="INR",
                status=PaymentStatus.SUCCESS,
                method=random.choice(["upi", "card", "netbanking"]),
                created_at=journey_time,
                updated_at=journey_time
            )
            payments_list.append(payment)

            # Event 6: PAYMENT_SUCCESS
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="PAYMENT_SUCCESS",
                entity_type="order",
                entity_id=str(order.id),
                event_metadata={"payment_id": fake_razorpay_payment_id, "amount": str(order_total), "method": payment.method},
                created_at=journey_time
            ))

            # Event 7: ORDER_CREATED
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="ORDER_CREATED",
                entity_type="order",
                entity_id=str(order.id),
                event_metadata={"order_id": str(order.id), "total_amount": str(order_total), "items_count": 1},
                created_at=journey_time
            ))

            # Rare return event (3 out of 25 converted customers returned an item)
            if converted_customers.index(customer) < 3:
                return_time = journey_time + timedelta(days=random.randint(3, 7))
                events_list.append(Event(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    customer_id=customer.id,
                    event_type="ORDER_RETURNED",
                    entity_type="order",
                    entity_id=str(order.id),
                    event_metadata={"reason": random.choice(["Size did not fit", "Changed preference", "Defective seam"]), "refund_initiated": True},
                    created_at=return_time
                ))

        # -------------------------------------------------------------
        # Segment B: Abandoned Cart Customers (15)
        # -------------------------------------------------------------
        for customer in abandoned_customers:
            journey_time = customer.created_at + timedelta(hours=random.randint(1, 24))
            search_term = random.choice(SEARCH_QUERIES)

            # Event: SEARCH
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="SEARCH",
                entity_type="search",
                entity_id=None,
                event_metadata={"query": search_term, "results_count": random.randint(10, 40)},
                created_at=journey_time
            ))

            viewed_p = random.choice(products_list)
            journey_time += timedelta(minutes=random.randint(2, 6))

            # Event: PRODUCT_VIEWED
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="PRODUCT_VIEWED",
                entity_type="product",
                entity_id=str(viewed_p.id),
                event_metadata={"product_title": viewed_p.title, "price": str(viewed_p.price)},
                created_at=journey_time
            ))

            prod_variants = [v for v in variants_list if v.product_id == viewed_p.id]
            selected_variant = random.choice(prod_variants) if prod_variants else random.choice(variants_list)

            # Cart (ABANDONED)
            cart = Cart(
                id=uuid.uuid4(),
                customer_id=customer.id,
                status=CartStatus.ABANDONED,
                created_at=journey_time,
                updated_at=journey_time + timedelta(hours=2)
            )
            carts_list.append(cart)

            cart_item = CartItem(
                id=uuid.uuid4(),
                cart_id=cart.id,
                variant_id=selected_variant.id,
                quantity=1,
                price_at_addition=selected_variant.price,
                created_at=journey_time,
                updated_at=journey_time
            )
            cart_items_list.append(cart_item)

            journey_time += timedelta(minutes=1)
            # Event: ADD_TO_CART
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="ADD_TO_CART",
                entity_type="cart",
                entity_id=str(cart.id),
                event_metadata={"variant_id": str(selected_variant.id), "sku": selected_variant.sku, "price": str(selected_variant.price)},
                created_at=journey_time
            ))

            # Some attempted checkout and payment failed
            if abandoned_customers.index(customer) < 6:
                journey_time += timedelta(minutes=random.randint(2, 5))
                # Event: CHECKOUT_STARTED
                events_list.append(Event(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    customer_id=customer.id,
                    event_type="CHECKOUT_STARTED",
                    entity_type="cart",
                    entity_id=str(cart.id),
                    event_metadata={"cart_total": str(selected_variant.price)},
                    created_at=journey_time
                ))

                order = Order(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    customer_id=customer.id,
                    cart_id=cart.id,
                    total_amount=selected_variant.price,
                    status=OrderStatus.FAILED,
                    created_at=journey_time,
                    updated_at=journey_time
                )
                orders_list.append(order)

                order_item = OrderItem(
                    id=uuid.uuid4(),
                    order_id=order.id,
                    variant_id=selected_variant.id,
                    quantity=1,
                    price=selected_variant.price,
                    created_at=journey_time
                )
                order_items_list.append(order_item)

                fake_rzp_order = f"order_{uuid.uuid4().hex[:14]}"
                payment = Payment(
                    id=uuid.uuid4(),
                    order_id=order.id,
                    razorpay_order_id=fake_rzp_order,
                    razorpay_payment_id=None,
                    razorpay_signature=None,
                    amount=selected_variant.price,
                    currency="INR",
                    status=PaymentStatus.FAILED,
                    method="upi",
                    created_at=journey_time,
                    updated_at=journey_time
                )
                payments_list.append(payment)

                # Event: PAYMENT_FAILED
                events_list.append(Event(
                    id=uuid.uuid4(),
                    merchant_id=merchant.id,
                    customer_id=customer.id,
                    event_type="PAYMENT_FAILED",
                    entity_type="order",
                    entity_id=str(order.id),
                    event_metadata={"error_code": "BAD_REQUEST_ERROR", "reason": "Bank server timeout"},
                    created_at=journey_time
                ))

            # Event: CART_ABANDONED
            abandon_time = journey_time + timedelta(hours=random.randint(1, 3))
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="CART_ABANDONED",
                entity_type="cart",
                entity_id=str(cart.id),
                event_metadata={"cart_value": str(selected_variant.price), "items_count": 1, "last_activity_mins_ago": 60},
                created_at=abandon_time
            ))

        # -------------------------------------------------------------
        # Segment C: Active Browsing Customers (10)
        # -------------------------------------------------------------
        for customer in active_customers:
            journey_time = now - timedelta(hours=random.randint(1, 6), minutes=random.randint(0, 50))

            # Event: SEARCH
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="SEARCH",
                entity_type="search",
                entity_id=None,
                event_metadata={"query": random.choice(SEARCH_QUERIES)},
                created_at=journey_time
            ))

            active_product = random.choice(products_list)
            journey_time += timedelta(minutes=random.randint(2, 5))

            # Event: PRODUCT_VIEWED
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="PRODUCT_VIEWED",
                entity_type="product",
                entity_id=str(active_product.id),
                event_metadata={"product_title": active_product.title, "price": str(active_product.price)},
                created_at=journey_time
            ))

            prod_variants = [v for v in variants_list if v.product_id == active_product.id]
            selected_variant = random.choice(prod_variants) if prod_variants else random.choice(variants_list)

            # Cart (ACTIVE)
            cart = Cart(
                id=uuid.uuid4(),
                customer_id=customer.id,
                status=CartStatus.ACTIVE,
                created_at=journey_time,
                updated_at=journey_time
            )
            carts_list.append(cart)

            cart_item = CartItem(
                id=uuid.uuid4(),
                cart_id=cart.id,
                variant_id=selected_variant.id,
                quantity=1,
                price_at_addition=selected_variant.price,
                created_at=journey_time,
                updated_at=journey_time
            )
            cart_items_list.append(cart_item)

            journey_time += timedelta(minutes=1)
            # Event: ADD_TO_CART
            events_list.append(Event(
                id=uuid.uuid4(),
                merchant_id=merchant.id,
                customer_id=customer.id,
                event_type="ADD_TO_CART",
                entity_type="cart",
                entity_id=str(cart.id),
                event_metadata={"variant_id": str(selected_variant.id), "sku": selected_variant.sku, "price": str(selected_variant.price)},
                created_at=journey_time
            ))

        # Save all generated entities
        db.add_all(carts_list)
        db.flush()
        db.add_all(cart_items_list)
        db.flush()
        db.add_all(orders_list)
        db.flush()
        db.add_all(order_items_list)
        db.flush()
        db.add_all(payments_list)
        db.flush()
        db.add_all(events_list)
        db.commit()

        # -------------------------------------------------------------
        # 5. Print Detailed Summary
        # -------------------------------------------------------------
        print("\n" + "=" * 55)
        print("🎉 SEEDING COMPLETED SUCCESSFULLY!")
        print("=" * 55)
        print(f"📊 SUMMARY OF GENERATED DATA:")
        print(f"  • Merchants      : {1}")
        print(f"  • Products       : {len(products_list)}")
        print(f"  • Variants       : {len(variants_list)}")
        print(f"  • Inventories    : {len(inventories_list)}")
        print(f"  • Customers      : {len(customers_list)}")
        print(f"  • Carts          : {len(carts_list)} ({sum(1 for c in carts_list if c.status == CartStatus.CONVERTED)} converted, {sum(1 for c in carts_list if c.status == CartStatus.ABANDONED)} abandoned, {sum(1 for c in carts_list if c.status == CartStatus.ACTIVE)} active)")
        print(f"  • Cart Items     : {len(cart_items_list)}")
        print(f"  • Orders         : {len(orders_list)} ({sum(1 for o in orders_list if o.status == OrderStatus.PAID)} paid, {sum(1 for o in orders_list if o.status == OrderStatus.FAILED)} failed)")
        print(f"  • Order Items    : {len(order_items_list)}")
        print(f"  • Payments       : {len(payments_list)} ({sum(1 for p in payments_list if p.status == PaymentStatus.SUCCESS)} success, {sum(1 for p in payments_list if p.status == PaymentStatus.FAILED)} failed)")
        print(f"  • Customer Events: {len(events_list)}")
        print("=" * 55)

    except Exception as exc:
        print(f"❌ Error during database seeding: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed Growearn / MerchantAI database.")
    parser.add_argument("--reset", action="store_true", default=True, help="Reset existing demo merchant data before seeding (default: True).")
    parser.add_argument("--no-reset", dest="reset", action="store_false", help="Do not reset existing demo merchant data.")
    args = parser.parse_args()

    seed_database(reset=args.reset)
