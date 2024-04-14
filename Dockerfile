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
RUN mv /home/steve/NMRcraft/.zshrc /root/.zshrc
RUN mv /home/steve/NMRcraft/.p10k.zsh /root/.p10k.zsh
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /root/powerlevel10k
RUN echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>/root/.zshrc
VOLUME [ "/home/steve" ]

# start a zsh shell
CMD ["zsh"]
