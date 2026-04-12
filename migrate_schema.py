#!/usr/bin/env python3
"""
Quick migration script to apply the schema changes for job_recommendation_requests.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from src.db.db import apply_schema
    
    print("Applying schema migration...")
    apply_schema()
    print("✓ Schema migration complete!")
    print("\nYou should now be able to create recommendation requests without constraint errors.")
    
except Exception as e:
    print(f"✗ Error applying schema: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
