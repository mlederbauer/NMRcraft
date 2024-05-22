FROM archlinux

# Install python, poetry and looks stuff from arch system repos
RUN pacman-key --init
RUN pacman-key --populate
RUN pacman -Syu python python-poetry ranger neovim eza git tree zsh openssh which neofetch github-cli make binutils gcc pkg-config fakeroot debugedit --noconfirm

# Set working directory and copy over config files and install python packages
RUN mkdir -p /home/steve
ADD https://api.github.com/repos/mlederbauer/NMRcraft/git/refs/heads/main version.json
RUN git clone https://github.com/mlederbauer/NMRcraft.git /home/steve/NMRcraft
WORKDIR /home/steve/NMRcraft
RUN echo "🚀 Creating virtual environment using pyenv and poetry"
#RUN poetry install
#RUN poetry run pre-commit install

# Get Arial font
RUN git clone https://aur.archlinux.org/ttf-ms-fonts.git
RUN cd ttf-ms-fonts
RUN chmod 777 .
RUN runuser -unobody makepkg
RUN pacman -U ttf-ms-fonts-2.0-12-any.pkg.tar.zst --noconfirm
RUN cd ..
RUN rm -r ttf-ms-fonts

# Quality of Life stuff
ADD https://api.github.com/repos/tiaguinho-code/Archpy_dots/git/refs/heads/main version.json
RUN git clone https://github.com/tiaguinho-code/Archpy_dots /home/steve/Archpy_dots
RUN chsh -s $(which zsh)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN cp /home/steve/Archpy_dots/.zshrc /root/.zshrc
VOLUME [ "/home/steve" ]

# start a zsh shell
CMD ["zsh"]
