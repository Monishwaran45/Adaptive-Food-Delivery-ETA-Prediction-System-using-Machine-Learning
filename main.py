"""
Launcher and Entry Point for the Adaptive Food Delivery ETA Prediction System
"""

import sys
import subprocess
import os


def main():
    print("=" * 70)
    print("🚀 Starting Adaptive Food Delivery ETA & Delay Risk Intelligence System")
    print("=" * 70)
    
    # Path to streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Launch Streamlit
    print(f"Launching Streamlit application from: {app_path}")
    print("Command: streamlit run app.py")
    print("-" * 70)
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting Streamlit.")


if __name__ == "__main__":
    main()
