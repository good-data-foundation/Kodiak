# ML-SDK
ML-SDK is use to training the model in multiple Data Owners, and consolidate
the result to a Query Customer.

## Installation
```bash
python setup.py sdist
cd ./dist
pip install *.tar.gz
```

## Requirement
Please install all the dependence packages first.
```bash
pip install -r requirements.txt
```

## install pre-commit
Please add ```.pre-commit-config.yaml``` at the root dirctory of this project
```bash
repos:
-   repo: https://github.com/pre-commit/mirrors-pylint
    rev: v2.5.3
    hooks:
    -   id: pylint
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.782
    hooks:
    -   id: mypy
```
After that run
```bash
pre-commit install
```

## License

The ml-sdk is licensed under the [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.txt), also included in our repository in the `LICENSE` file.
