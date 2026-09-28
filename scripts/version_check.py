"""Check installed package versions for reproducibility."""
import sys


PACKAGES = [
    "numpy", "pandas", "rasterio", "geopandas", "matplotlib",
    "seaborn", "scipy", "sklearn", "SALib", "libpysal", "esda",
    "yaml", "tqdm",
]


def main():
    print("=" * 60)
    print("REPRODUCIBILITY VERSION CHECK")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print()
    for pkg in PACKAGES:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "N/A")
            print(f"  {pkg:<15}: {ver}")
        except ImportError:
            print(f"  {pkg:<15}: NOT INSTALLED")
    print("=" * 60)


if __name__ == "__main__":
    main()