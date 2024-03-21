FROM archlinux

# Install python, poetry and looks stuff from arch system repos
RUN pacman -Syu python python-poetry ranger neovim eza git tree zsh openssh which neofetch --noconfirm

# Set working directory and copy over config files and instl
WORKDIR /NMRcraft
COPY poetry.toml pyproject.toml ./
RUN poetry install

# Quality of Life stuff
RUN chsh -s $(which zsh)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY .zshrc /root/.zshrc

VOLUME [ "/NMRcraft" ]

# start a zsh shell
CMD ["zsh"]
