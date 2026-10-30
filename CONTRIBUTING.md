# Contributing

Thanks for taking a look. The bar for changes is the same as for the code
already here: typed, tested, and explained.

1. `make install` then `make check` must pass (ruff, `mypy --strict`, pytest).
2. New checks and mutators need a unit test, and a line in the README tables.
3. Behaviour changes that alter what the example suite reports should update
   `tests/test_example_suite.py` in the same PR, with the reason in the description.
4. Keep runtime dependencies at four. Provider SDKs belong in optional extras.
5. One logical change per commit, written in the imperative ("Add X", "Fix Y").

Bug reports are most useful with the suite YAML, the provider spec, the seed,
and the relevant part of the run JSON.
