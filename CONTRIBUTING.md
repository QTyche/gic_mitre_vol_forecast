# Contributing

Use a focused branch and keep each change small enough to test independently.
Run the following before opening a review:

```bash
make test
make lint
make smoke
```

Scientific changes must include the configuration that activates them and must
describe their effect on the data contract, temporal split, random seeds, and
reported metrics. Generated result files are not source code; retain only
reviewed, reproducible submission artifacts when the project reaches that stage.

