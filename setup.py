from setuptools import setup, find_packages

setup(
    name="qx_ir",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'click>=8.0.0',
    ],
    entry_points={
        'console_scripts': [
            'qx=qx_ir.cli.main:qx',
        ],
    },
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum SDK for quantum circuit execution and analysis",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zenaquantum",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
