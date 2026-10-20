Developer Guide
===============

Please adhere to the following guidelines and explanations when contributing to this project.

Environment setup
-----------------

Clone the latest development version of the repository with the following command::

    git clone --recurse-submodules -b dev https://github.com/3dct/vIQA

It's best practice to use a virtual environment for development. For example you can create a new virtual environment with the following command::

    python -m venv /path/to/myenv

To set up the development environment, you need to install the dependencies for the project. You can do this by running the following command after cloning the repository::

    pip install -e .[dev]

This will install the project in editable mode and install the development dependencies.

Running tests
-------------

You can run the tests with the following command from the root of the repository::

    pytest

Commit guidelines
-----------------

We use `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ for our commit messages. For in-depth information, please look there. A short summary is provided below.
This is important because we use `python-semantic-release <https://python-semantic-release.readthedocs.io/en/latest/>`_ to automate the versioning and release process.
Accepted commit types are:

*   ``feat``: A new feature
*   ``fix``: A bug fix
*   ``docs``: Documentation only changes
*   ``style``: Changes that do not affect the meaning of the code (whitespace, formatting, missing semicolons, etc)
*   ``refactor``: A code change that neither fixes a bug nor adds a feature
*   ``perf``: A code change that improves performance
*   ``test``: Adding missing tests or correcting existing tests
*   ``build``: Changes that affect the build system or external dependencies (example scopes: pip)
*   ``ci``: Changes to our CI configuration files and scripts (example scopes: GitHub Actions)
*   ``chore``: Changes that affect the chore of the project (example scopes: versioning, release, configs, etc)
*   ``book``: Changes to the jupyter books

You can use a scope to specify the part of the codebase that is affected by the commit if you want to.

For example, ``feat(api): add new endpoint`` or ``fix(api.py): fix bug in endpoint``.

Use the imperative, present tense: "change" not "changed" nor "changes" in the commit message.

Don't capitalize the first letter of the commit message and limit the first line to 72 characters or less.

Add a blank line after the first line to add a more detailed description if needed.

Add a reference to the issue that the commit closes if applicable.

Use ``BREAKING CHANGE:`` at the start of the commit message if the commit introduces a breaking change and add a exclamation mark (!) after the commit type.
For example::

    feat!: add new endpoint

    BREAKING CHANGE: this commit breaks the API

We also use Atomic Commits. This means that each commit should only contain changes related to one feature or bug fix. There's nothing wrong with having multiple commits
for a single feature or bug fix or committing only a few lines of code. This makes it easier to understand the history of the project and to revert changes if necessary.

Code style
----------

We use `ruff <https://docs.astral.sh/ruff/>`_ to enforce a consistent code style across the project. Please have a look at the pyproject.toml file for the configuration.
Code should be checked with ``ruff check`` and formatted with ``ruff format`` before committing. Though ci will run these checks as well, it is good practice to run them locally before committing.
Several linters are implemented via pre-commit hooks. Please install pre-commit and run ``pre-commit install`` to enable them. You can run the linters with ``pre-commit run --all-files`` manually. This includes
the ruff checks and formatting as well.

Documentation
-------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate our API reference. Please add documentation to all new public features and changes via docstrings in the code.
`NumPy <https://numpydoc.readthedocs.io/en/latest/format.html>`_ style docstrings must be used. Additionally .rst files are used for more detailed documentation.
If you installed the dev dependencies, you can build the documentation with the following command::

    sphinx-build -b html docs/source docs/build

Alternatively, if you want to rebuild the documentation automatically on changes you can use the following command::

    sphinx-autobuild --open-browser docs/source docs/build/html --watch src/viqa

The documentation will be available in the ``docs/build`` directory. If you build the documentation multiple times, you may have to delete the ``docs/build`` directory before building again.
This is for example possible with::

    cd docs
    make clean

Versioning and Dependencies
---------------------------

Python Semantic Release is used to automate the versioning and release process. This follows the `Semantic Versioning <https://semver.org/>`_ guidelines.
Therefore the version dunder does not need to be updated manually. If you want to update the version manually anyway please use a dev version (e.g. 0.1.0.dev0) and tag the commit accordingly (e.g. v0.1.0.dev0).
Please do this only on the dev branch. The version will be updated automatically on the main branch.
PEP 440 versioning is used in the newest spec of `PyPA <https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers>`_.
NEP 29 is followed for the project. Please have a look at the `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ for more information.

Pre-commit usage
----------------

This repository uses `pre-commit <https://pre-commit.com/>`_ to manage the hooks. Please install pre-commit with the following command::

    pip install pre-commit

To install pre-commit hooks (including the pre-push hook to check if the docs can be built), run the following command::

    pre-commit install --hook-type pre-commit --hook-type pre-push

You can now run the pre-commit hooks manually with the following command::

    pre-commit run --all-files --hook-stage pre-commit --hook-stage pre-push
