FROM jupyter/minimal-notebook

# Package stuff: Install poetry with pip, install everything else with poetry
RUN pip install --no-cache-dir poetry
WORKDIR /code
COPY poetry.toml pyproject.toml ./
RUN poetry install

WORKDIR /code

COPY . .
VOLUME [ "/code/gen-data" ]

# Launch the Jupyter environment

CMD [ "bash" ]