FROM root/app:18.04

# remove unconvenient id (ONLY FOR MACOS)
USER 0
RUN groupmod -g 200 dialout
RUN find / -gid 20 ! -type l -exec chgrp 200 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }
RUN groupmod -g 20 staff && \
    find / -gid 50 ! -type l -exec chgrp 20 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }

# install package for mac osx socket sharing through x11
RUN apt-get install -y libglu1-mesa

# add new sudo user (replace username, groupname, 1000 (userid) and 1001 (groupid) by your values)
ENV USERNAME username
ENV GROUPNAME groupname
ENV UID 1000
ENV GID 1001
RUN useradd -N -m $USERNAME --uid $UID --gid $GID && \
    echo "$USERNAME:$GROUPNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
