"""
Redirect file to launch the Streamlit app properly.
This file exists only to satisfy systems that might be looking for a main.py entry point.
"""
import os
import sys

# Add the current directory to the path so we can import the app module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the app module and run it
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    from app import main
    
    # Try to run the Streamlit app directly
    sys.argv = ["streamlit", "run", "app.py"]
    stcli.main()