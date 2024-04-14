FROM archlinux

# Install python, poetry and looks stuff from arch system repos
RUN pacman -Syu python python-poetry ranger neovim eza git tree zsh openssh which neofetch github-cli --noconfirm

# Set working directory and copy over config files and instl
RUN mkdir -p /home/steve
WORKDIR /home/steve/NMRcraft
RUN git clone https://github.com/mlederbauer/NMRcraft.git /home/steve/NMRcraft
RUN poetry install

# Quality of Life stuff
RUN chsh -s $(which zsh)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN cp /home/steve/NMRcraft/.zshrc /root/.zshrc
VOLUME [ "/home/steve" ]

# start a zsh shell
CMD ["zsh"]
