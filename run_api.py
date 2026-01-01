#!/usr/bin/env python3
"""
Run the FastAPI server locally
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting NBA Predictions API Server")
    print("=" * 60)
    print("\nServer will be available at:")
    print("  http://localhost:8000")
    print("\nAPI docs at:")
    print("  http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
