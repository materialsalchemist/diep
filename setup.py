from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).resolve().parent
README = (here / "README.md").read_text(encoding="utf-8")
VERSION = (here / "src" / "VERSION").read_text(encoding="utf-8").strip()

setup(
    name="diep",
    packages=[
        "src/diep",
    ]
    + find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": ["diep=diep.cli:execute_cli"],
    },
    version=VERSION,
    license="mit",
    description="material representation via the direct integration of the external potential",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Sherif Abdulkader Tawfik Abbas, Tri ",
    author_email="sherif.tawfic@gmail.com",
    url="https://github.com/sheriftawfikabbas/diep",
    keywords=["material science", "graph neural networks", "ai", "machine learning"],
    install_requires=[
        "ase",
        "dgl",
        "pytorch_lightning",
        "pymatgen",
        "m3gnet",
        "boto3",
    ],
)
