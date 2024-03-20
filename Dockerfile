FROM python

# Package stuff: Install poetry with pip, install everything else with poetry
RUN pip install --no-cache-dir poetry
WORKDIR /code
COPY poetry.toml pyproject.toml ./
RUN poetry install

WORKDIR /code

COPY ./nmrcraft/ ./nmrcraft/
VOLUME [ "/code/gen-data" ]
COPY ./tests ./tests

# Use the python binary of the local environment
CMD ["poetry", "run", "python", "nmrcraft/main.py"]
