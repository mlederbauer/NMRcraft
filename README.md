![nmrcraft_logo](assets/NMRCRAFT-logo.png)

[![Release](https://img.shields.io/github/v/release/mlederbauer/nmrcraft)](https://img.shields.io/github/v/release/mlederbauer/nmrcraft)
[![Build status](https://img.shields.io/github/actions/workflow/status/mlederbauer/nmrcraft/main.yml?branch=main)](https://github.com/mlederbauer/nmrcraft/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/mlederbauer/nmrcraft/branch/main/graph/badge.svg)](https://codecov.io/gh/mlederbauer/nmrcraft)
[![Commit activity](https://img.shields.io/github/commit-activity/m/mlederbauer/nmrcraft)](https://img.shields.io/github/commit-activity/m/mlederbauer/nmrcraft)
[![License](https://img.shields.io/github/license/mlederbauer/nmrcraft)](https://img.shields.io/github/license/mlederbauer/nmrcraft)

<h1 align="center">
  NMRcraft
</h1>
<h2 align="center">
  NMR Chemical Reactivity Analysis with Feature Tracking
</h2>
<p align="center">
NMRcraft will help you to mine into your NMR data and craft awesome predictions!
</p>

- **Github repository**: <https://github.com/mlederbauer/nmrcraft/>
- **Documentation** <https://mlederbauer.github.io/nmrcraft/>

# 🔥 Usage

## Docker Desktop 🐳

First you need to install [Docker](https://www.docker.com/products/docker-desktop/).

### Download Docker Image

You can download the image by going onto the searchbar on top and searching for 'tiaguinho/nmrcraft_arch' and clicking on pull.

### Running the Image

To run the image you need to go to the 'Images' tab and click the "play" button on the nmrcraft*arch container you pulled. It should appear as running in the 'Containers' tab and there you should click on the ⋮ symbol and click on '>* open in termnial'. After that a terminal window should pop up where you will type in the command `zsh`.

## Console 🐧

### Download Docker Image

To use the docker image just pull it from [Docker Hub](https://hub.docker.com/r/tiaguinho/nmrcraft_arch) and make sure [Docker](https://www.docker.com/products/docker-desktop/) is installed. To pull it you can execute this command:

```bash
docker pull tiaguinho/nmrcraft_arch
```

(If you're o windows you might need to call docker.exe instead of just docker)

### Running the Image

```bash
docker run -it nmrcraft_arch
```

## Visual Studio Code 🪟

To download the image follow the same steps as either console or docker desktop.

### Running the Docker Image

To run follow the following tutorial on how to get Docker to work nicely with VS Code.

<details>
<summary>Using Docker in VS Code</summary>
<ol>
<li> Open VS Code and install the extensions for Docker and Dev Containers.</li>
<li> Go to the newly added Docker Tab. Here you should now see three sections: Containers, Images and Registries. And under Images the tiaguinho/nmrcraft_arch image should be visible.</li>
<li> In order for the container not to be deleted every time you stop it we have to remove the --rm commad. For this go to the settings (Ctrl + , on Mac) and type `docker run`. Select 'Edit the settings.json' for the 'Run Interactive' command and remove the --rm to get: "docker.commands.runInteractive": "${containerCommand} run -it ${exposedPorts} ${tag}", "docker.commands.run": "${containerCommand} run -d ${exposedPorts} ${tag}". Save the file.</li>
<li> In the Docker Tab on the right, right click on the image and select run interactive. Now a conainer should appear in the Container section. Right click on it and select stop to start it back up.</li>
<li> Right click again on the container and select start to start it back up.</li>
<li> Right click again on the container and select attach Visual Studio Code. A new VS Code window should apear, this window is now fully in the container. If necessary, switch to `/home/steve/NMRcraft`.</li>
<li>Pull the latest changes to the repository with `git pull origin main`.</li>
<li> Have fun developing.</li>
</ol>
</details>

## Getting Access to the Dataset 💾

For the script to be able to access the dataset, you must login via to huggingface by using the following command:

```bash
huggingface-cli login
```

# 👩‍💻 App

Run the app demo locally with

```bash
python nmrcraft/app.py
```

# 🖼️Poster

WIP #TODO

# 🧑‍💻 Developing

### Activate the Poetry venv

To use the packages installed via poetry you need to execute the following command:

```bash
poetry shell
```

This will put you into the poetry shell from where you have direct access to all packages managed by poetry.

### GitHub pushing auth

To authenticate the Docker comes with the github cli application. To login execute this command:

```bash
gh auth login
```

and follow the interactive instructions with enter and the arrow keys. Once logged in you should be able to push changes to the repo.

### Adding packages and libraries to the project

If you added a new feature that requires a new package/library, you can add by running `poetry add <package-name>` and run `make install` to install the new dependencies.

(You might need to run `poetry lock` to update the `poetry.lock` file if you added a dependency manually in the `pyproject.toml` file.)

### Loading the Data

The dataset is stored in a private repository on HuggingFace.

To download the dataset on the Hub in Python, you need to log in to your Hugging Face account:

```bash
huggingface-cli login
```

Access the dataset:

```python
from nmrcraft.data.data_utils import load_dataset_from_hf
load_dataset_from_hf()
```

The raw dataset can then be found in 'dataset/dataset.csv'

### References

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
