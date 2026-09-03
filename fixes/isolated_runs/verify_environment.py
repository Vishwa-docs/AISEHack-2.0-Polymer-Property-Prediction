"""Fail fast when the isolated notebook environment drifts from its required pins."""
from __future__ import annotations

import sys

EXPECTED = {
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "sklearn": "1.9.0",
    "rdkit": "2026.03.5",
    "lightgbm": "4.7.0",
}


def main() -> None:
    assert sys.version_info[:3] == (3, 11, 7), sys.version
    import lightgbm
    import numpy
    import pandas
    import rdkit
    import sklearn

    actual = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "lightgbm": lightgbm.__version__,
    }
    mismatch = {key: (EXPECTED[key], actual[key]) for key in EXPECTED if actual[key] != EXPECTED[key]}
    if mismatch:
        raise SystemExit(f"Pin mismatch: {mismatch}")
    print("ENVIRONMENT OK")
    print("Python", sys.version.split()[0])
    for key in EXPECTED:
        print(f"{key} {actual[key]}")


if __name__ == "__main__":
    main()
