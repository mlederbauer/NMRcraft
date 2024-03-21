FROM python

# Package stuff: Install poetry with pip, install everything else with poetry
RUN pip install --no-cache-dir poetry
WORKDIR /NMRcraft
COPY poetry.toml pyproject.toml ./
RUN poetry install


VOLUME [ "/NMRcraft" ]

# Use the python binary of the local environment
CMD ["bash"]
