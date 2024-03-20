FROM python

# Package stuff: Install poetry with pip, install everything else with poetry
RUN pip install --no-cache-dir poetry
WORKDIR /code
COPY poetry.toml pyproject.toml ./
RUN poetry install

WORKDIR /NMRcraft

VOLUME [ "/NMRcraft" ]
COPY ./tests ./tests

# Use the python binary of the local environment
CMD ["bash"]
