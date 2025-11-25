from importlib.metadata import version, PackageNotFoundError

try:
        # Distribution name on PyPI / in the wheel
            __version__ = version("lammps-trajan")
except PackageNotFoundError:
        # Editable or source tree – version unknown
            __version__ = "0.0.dev0"

