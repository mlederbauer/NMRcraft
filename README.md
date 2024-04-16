![nmrcraft_logo](assets/NMRCRAFT-logo.png)

[![Release](https://img.shields.io/github/v/release/mlederbauer/nmrcraft)](https://img.shields.io/github/v/release/mlederbauer/nmrcraft)
[![Build status](https://img.shields.io/github/actions/workflow/status/mlederbauer/nmrcraft/main.yml?branch=main)](https://github.com/mlederbauer/nmrcraft/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/mlederbauer/nmrcraft/branch/main/graph/badge.svg)](https://codecov.io/gh/mlederbauer/nmrcraft)
[![Commit activity](https://img.shields.io/github/commit-activity/m/mlederbauer/nmrcraft)](https://img.shields.io/github/commit-activity/m/mlederbauer/nmrcraft)
[![License](https://img.shields.io/github/license/mlederbauer/nmrcraft)](https://img.shields.io/github/license/mlederbauer/nmrcraft)

<h1 align="center">
  NMRcraft
</h1>

NMR Chemical Reactivity Analysis with Feature Tracking

INSERT CATCHPHRASE #TODO
How to ... ?
NMRcraft leverages ...

- **Github repository**: <https://github.com/mlederbauer/nmrcraft/>
- **Documentation** <https://mlederbauer.github.io/nmrcraft/>

## 🔥 Usage

WIP #TODO

## 👩‍💻 App

Run the app demo locally with

```bash
python nmrcraft/app.py
```

## 🖼️Poster

WIP #TODO

## 🧑‍💻 Developing

### Installation

To develop this project, start the environment by either

```bash
make install
```

to update all the packages in case you added any poetry packages for testing or just

```bash
poetry shell
```

if you just pulled this docker image.

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

### Adding dependencies to the project

If you added a new feature that requires a new package/dependency, you can add it to the `pyproject.toml` file and run `make install` to install the new dependencies.

```bash
poetry add <package-name>
```

(You might need to run `poetry lock` to update the `poetry.lock` file if you added a dependency manually in the `pyproject.toml` file.)

### Loading the Data

The dataset is stored in a private repository on HuggingFace.

To download the dataset on the Hub in Python, you need to log in to your Hugging Face account:

```bash
huggingface-cli login
```

Huggingface is already installed via poetry, so make sure you either entered the command `make install`or you entered `poetry shell`, so you are in the poetry environment.

Access the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("NMRcraft/nmrcraft", data_files='all_no_nan.csv')
dataset['train'].to_pandas() # contains the data for now
```

Or download locally in just a few lines:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="NMRcraft/nmrcraft", filename="all_no_nan.csv", repo_type="dataset", local_dir="./data/")

```

The dataset is provided in its "raw" form as a dataframe in `all_no_nan.csv`, meaning without a train/test split or feature selection.
in the folder `./xyz.zip`, all optimized geometries are added as an .xtpopt.xyz file.

### Using Docker

To use the docker image just pull it from [Docker Hub](https://hub.docker.com/r/tiaguinho/nmrcraft_arch)

<details>
<summary>Docker in VS Code</summary>
<ol>
<li> Docker Desktop has to be installed for your specific operating system. (https://www.docker.com/products/docker-desktop/)</li>
<li> Open a new shell and download the NMRcraft image called tiaguinho/nmrcraft_arch with the command `docker pull tiaguinho/nmrcraft_arch`</li>
<li> Open VS Code and install the extensions for Docker and Dev Containers.</li>
<li> Go to the newly added Docker Tab. Here you should now see three sections: Containers, Images and Registries. And under Images the tiaguinho/nmrcraft_arch image should be visible.</li>
<li> In order for the container not to be deleted every time you stop it we have to remove the --rm commad. For this go to the settings and type docker run. Select 'Edit the settings.jason' for the 'Run Interactive' command and remove the --rm to get: "docker.commands.runInteractive": "${containerCommand} run -it ${exposedPorts} ${tag}", "docker.commands.run": "${containerCommand} run -d ${exposedPorts} ${tag}". Save the file.</li>
<li> In the Docker Tab on the right, right click on the image and select run interactive. Now a conainer should appear in the Container section. Right click on it and select stop to start it back up.</li>
<li> Right click again on the container and select start to start it back up.</li>
<li> Right click again on the container and select attach Visual Studio Code. A new VS Code window should apear, this window is now fully in the container.</li>
<li> Have fun developing.</li>
</ol>
</details>

### Building a Docker Image and running it manually

To build a Docker image you can run the following commands on _Linux/MacOS_ to build and run an image:

```bash
docker buildx build -t pypy .
```

```bash
docker run -v ./.:/NMRcraft -it pypy
```

On Windows powershell, the commands are:

```bash
docker.exe buildx build -t pypy .
```

```bash
docker.exe run -v ./.:/NMRcraft -it pypy
```
