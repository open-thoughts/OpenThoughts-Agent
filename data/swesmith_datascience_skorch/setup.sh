#!/bin/bash 

# Clone your SWE-smith fork if it doesn't exist
if [ ! -d "$HOME/github.com/nick11roberts/SWE-smith" ]; then
    git clone https://github.com/nick11roberts/SWE-smith.git $HOME/github.com/nick11roberts/SWE-smith
    cd $HOME/github.com/nick11roberts/SWE-smith
    git checkout data-science-repos
else
    echo "SWE-smith directory already exists, updating..."
    cd $HOME/github.com/nick11roberts/SWE-smith
    git pull
    git checkout data-science-repos
fi

# Install your SWE-smith fork in development mode
cd $HOME/github.com/nick11roberts/SWE-smith
pip install -e .

echo "Setup completed successfully!"

