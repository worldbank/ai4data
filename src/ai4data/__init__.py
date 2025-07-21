from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai4data")
except PackageNotFoundError:
    # package is not installed
    pass
