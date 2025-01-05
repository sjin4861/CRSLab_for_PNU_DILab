from setuptools import setup, find_packages

try:
    import torch
    import torch_geometric
except Exception:
    raise Exception('Please install PyTorch and PyTorch Geometric manually first.\n' +
                    'View CRSLab GitHub page for more information: https://github.com/RUCAIBox/CRSLab')

setup_requires = []

install_requires = [
    'absl-py==2.1.0',
    'aiohappyeyeballs==2.4.4',
    'aiohttp==3.11.11',
    'aiosignal==1.3.2',
    'attrs==24.3.0',
    'certifi==2024.12.14',
    'charset-normalizer==3.4.1',
    'click==8.1.8',
    'colorama==0.4.6',
    'Cython==3.0.11',
    'dataclasses==0.6',
    'filelock==3.16.1',
    'frozenlist==1.5.0',
    'fsspec==2024.12.0',
    'fuzzywuzzy==0.18.0',
    'grpcio==1.69.0',
    'huggingface-hub==0.27.0',
    'idna==3.10',
    'Jinja2==3.1.5',
    'joblib==1.4.2',
    'loguru==0.7.3',
    'Markdown==3.7',
    'MarkupSafe==3.0.2',
    'mpmath==1.3.0',
    'multidict==6.1.0',
    'networkx==3.4.2',
    'nltk==3.9.1',
    'numpy==2.2.1',
    'packaging==24.2',
    'pillow==10.2.0',
    'pkuseg==0.0.25',
    'propcache==0.2.1',
    'protobuf==5.29.2',
    'psutil==6.1.1',
    'pyparsing==3.2.1',
    'PyYAML==6.0.2',
    'regex==2024.11.6',
    'requests==2.32.3',
    'safetensors==0.5.0',
    'scikit-learn==1.6.0',
    'scipy==1.15.0',
    'sentencepiece==0.2.0',
    'six==1.17.0',
    'sympy==1.13.1',
    'tensorboard==2.18.0',
    'tensorboard-data-server==0.7.2',
    'threadpoolctl==3.5.0',
    'tokenizers==0.21.0',
    'torch==2.5.1+cu124',
    'torch-geometric==2.6.1',
    'torchaudio==2.5.1+cu124',
    'torchvision==0.20.1+cu124',
    'tqdm==4.67.1',
    'transformers==4.47.1',
    'typing_extensions==4.12.2',
    'urllib3==2.3.0',
    'Werkzeug==3.1.3',
    'win32_setctime==1.2.0',
    'yarl==1.18.3'
]

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces"
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='crslab',
    version='0.1.1',  # please remember to edit crslab/__init__.py in response, once updating the version
    author='CRSLabTeam',
    author_email='francis_kun_zhou@163.com',
    description='An Open-Source Toolkit for Building Conversational Recommender System(CRS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RUCAIBox/CRSLab',
    packages=[
        package for package in find_packages()
        if package.startswith('crslab')
    ],
    classifiers=classifiers,
    install_requires=install_requires,
    setup_requires=setup_requires,
    python_requires='>=3.6',
)
