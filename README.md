Simplicial complexes for topological data analysis
===

[Pierre Sara](https://github.com/pierresara/) and [Ousema Bouaneni](https://github.com/ousema-bouaneni)

A program to calculate [Čech](https://en.wikipedia.org/wiki/%C4%8Cech_complex) and alpha complexes of a set of points in arbitrary dimension using an [LP-type] approach, done as part of [CSC_42021_EP]: an algorithm design course at École polytechnique and supervised by Marc Glisse.

# Project structure
If we ignore the `__init__.py` files that are there just to help python recognize the different modules of the package, our project's folder structure looks like this:

```
simplicial-complex/
├─ src/
│  ├─ project.py
├─ tests/
│  ├─ helper.py
│  ├─ circumcenter_test.py
│  ├─ meb_test.py
│  ├─ cech_complex_test.py
│  ├─ alpha_complex_test.py
├─ report/
│  ├─ rapport.pdf
│  ├─ annexe.pdf
README.md
```

- `src/project.py` is the main file. It contains functions to compute the minimum enclosing ball, the Cech complex and the alpha complex of a set of points in arbitrary dimension.
- The `tests/` directory contains files to test the functions defined in `src/project.py `, as well as a `helper.py` file with helper functions used across different test suites.
- The `reports/` directory contains two pdf documents written in french. The main theoretical and empirical results for the project are in `rapport.pdf` whereas `annexe.pdf` contains some additional mathematical proofs that were omitted to respect the page limit.

# Dependencies

In order to run `project.py`, only `numpy` is needed.
It can be installed using the command
```
pip install numpy
```
If you want to run the tests yourself, you also need to install `pytest` with the following command:
```
pip install pytest
```

# Running the tests

In order to run all tests, one needs to navigate to the project root and then run `pytest` as follows:
```
cd simplicial-complex
pytest -v
```
or equivalently:
```
cd simplicial-complex
python -m pytest -v
```
