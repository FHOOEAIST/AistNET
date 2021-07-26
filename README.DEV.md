# AistNET DEV documentation

## Run Test

To execute the test, `tox` is required to be installed in the system. Execute `tox` to trigger a full check with linting
and tests. To recreate the test environment because of some added package or changed version then execute `tox -r`.

## Build wheel package

`python -m build` this creates a wheel package in folder `dist` with the file `aistnet-0.0.1-py3-none-any.whl`. The
version number can change in the name.

## Manually install the wheel package

`pip install aistnet/aistnet-0.0.1-py3-none-any.whl`

Use the following flags for a more precise installation:

* `--no-deps` installs the package without the dependencies that are defined in the package
* `--force-reinstall` force a installation of the package, needed when a version with the same version is already
  installed
