# EV-Utils

EV-Utils is a personal python toolbox. It contains a range of utility functions for python development.

## Installation

Currently, you could install by run

```
git clone https://github.com/whuhangzhang/evutils.git
cd evutils
pip install -e .
```

## Code Style
We follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for code style.  Especially the style of docstrings is important to generate documentation.

* *Local*: Run the following commands in the package root directory
```
# Python syntax errors or undefined names
flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics
# Style checks
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
* *Github*: We use [Codacy](https://www.codacy.com) to check styles on pull requests and branches.