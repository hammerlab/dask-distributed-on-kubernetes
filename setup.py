try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

version = "0.0.1"

setup(
    name="mhcflurry-cloud",
    version=version,
    author="Tim O'Donnell",
    author_email="timodonnell@gmail.com",
    packages=["mhcflurry_cloud"],
    url="https://github.com/hammerlab/mhcflurry-cloud",
    license="Apache License",
    description="Cloud utils for mhcflurry",
    entry_points={
        'console_scripts': [
        ]
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
    ],
    install_requires=[
        "mhcflurry",
  #      "pepdata",
        "joblib",
        "nose>=1.3.1",
        "pandas>=0.16.1",
    ]
)
