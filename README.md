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
  Crafting Catalysts from NMR Features
</h2>
<p align="center">
NMRcraft is a project that predicts ligands of complexes from their chemical shift tensors.
</p>

# 🐳 Installation

<details>
  <summary>See installation instructions</summary>

## Docker Desktop 🐳

First you need to install [Docker](https://www.docker.com/products/docker-desktop/).

### Download Docker Image

You can download the image by going onto the searchbar on top and searching for 'tiaguinho/nmrcraft_arch' and clicking on pull.

### Running the Image

To run the image you need to go to the 'Images' tab and click the "play" button on the nmrcraft*arch container you pulled. It should appear as running in the 'Containers' tab and there you should click on the ⋮ symbol and click on '>* open in termnial'. After that a terminal window should pop up where you will type in the command `zsh`.

## Console 🐧

### Download Docker Image

To use the docker image, pull it from [Docker Hub](https://hub.docker.com/r/tiaguinho/nmrcraft_arch) and make sure that [Docker](https://www.docker.com/products/docker-desktop/) is installed. To pull it you can execute this command:

```bash
docker pull tiaguinho/nmrcraft_arch
```

(If running on windows, you might need to call docker.exe instead of just docker)

### Running the Image

```bash
docker run -it nmrcraft_arch
```

## Visual Studio Code 🪟

To download the image, follow the same steps as either console or docker desktop.

### Running the Docker Image

<details>
<summary>Using Docker in VS Code</summary>
<ol>
<li> Open VS Code and install the extensions for Docker and Dev Containers.</li>
<li> Go to the newly added Docker Tab. Here you should now see three sections: Containers, Images and Registries. And under Images the tiaguinho/nmrcraft_arch image should be visible.</li>
<li> In order for the container not to be deleted every time you stop it we have to remove the --rm commad. For this go to the settings (Ctrl + , on Mac) and type `docker run`. Select 'Edit the settings.json' for the 'Run Interactive' command and remove the --rm to get: "docker.commands.runInteractive": "${containerCommand} run -it ${exposedPorts} ${tag}", "docker.commands.run": "${containerCommand} run -d ${exposedPorts} ${tag}". Save the file.</li>
<li> In the Docker Tab on the right, right click on the image and select run interactive. Now a conainer should appear in the Container section. Right click on it and select stop to start it back up.</li>
<li> Right click again on the container and select start to start it back up.</li>
<li> Right click again on the container and select attach Visual Studio Code. A new VS Code window should apear, this window is now fully in the container. If necessary, switch to `/home/steve/NMRcraft`.</li>
<li> Pull the latest changes to the repository with `git pull origin main`.</li>
<li> Have fun developing.</li>
</ol>
</details>

## Getting Access to the Dataset 💾

For the script to be able to access the dataset, you must login via to huggingface by using the following command:

```bash
pip install -U "huggingface_hub[cli]" # if not installed already
huggingface-cli login # log in after generating an authentification token for huggingface
```

We include the link to be authenticated in the report appendix. If you run into issues accessing the dataset, contact [mlederbauer@ethz.ch](mlederbauer@ethz.ch).

</details>

# 🔥 Usage

To reproduce all results shown in the report, run the following commands:

```bash
poetry shell
python scripts/reproduce_results.py
```

This script will interatively

- plot dataset statistics and PCA plots (stored in `./plots/dataset`)
- train and evaluate all single-output models (stored in `./metrics/results_one_targets.csv`)
- train and evaluate all multi-output models (stored in `./metrics/results_multi_target.csv`)
- train and evaluate all baseline models (stored in `./metrics/results_baselines.csv`)
- create the plots (stored in `./plots/{models,baselines,dataset_statistics,results}`)
- print the table of experiment 3 to the terminal.

When the parameter `max_eval` is set to a high value such as 50, expect the whole process to take about two hours. Alternatively – which results in worse model performance –, `max_eval` can be set to a low value such as 2 for testing. Run `scripts/training/{one_target,multi_targets}.sh` for running individual pipelines (although running `scripts/reproduce_results.py` is recommended). Results are also accessible via the polybox [here](https://polybox.ethz.ch/index.php/s/CX9zH819uTlL4sr).

# 🖼️Poster

If you were not able to visit our beautiful poster at ETH Zurich on May 30th 2024, you can access our poster [here](assets/Poster.pdf)!

![Poster](assets/Poster_1000dpi.png)

# 🧑‍💻 Developing

<details>
  <summary>See developer instructions</summary>

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

</details>

# Citation

```
@software{nmrcraft2024,
  author       = {Magdalena Lederbauer and Karolina Biniek and Tiago Würthner and Samuel Stricker and Yingnan Wang},
  title        = {{mlederbauer/NMRcraft: Crafting Catalysts from NMR Features}},
  month        = may,
  year         = 2024
}
```

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
