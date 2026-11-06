import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import joblib


def main():
    print("CareCopilot environment check")
    print("-" * 40)

    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("SciPy:", scipy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Joblib:", joblib.__version__)

    print("-" * 40)
    print("CareCopilot Python environment is working correctly.")


if __name__ == "__main__":
    main()