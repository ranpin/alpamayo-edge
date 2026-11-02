import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_r1t import main
sys.exit(0 if main('https://ranpin.github.io/alpamayo-edge/r1-trajectory/') else 1)
