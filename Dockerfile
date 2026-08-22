# Base image specification. Defines the foundation OS and python version for the container (Required)
FROM openswe-python-3.11
# Fetch source code. same as git clone fonttools/fonttools && git reset --hard f1c609aa57fa11ab98f2152275f2c709e06c0680 but avoid network stuff; guarantee to ready
COPY repo /testbed
# set default workdir to testbed. (Required)
WORKDIR /testbed/
# The lines above should NEVER change (except python version), so as to reuse layers.

# Install system dependencies if needed
ENV DEBIAN_FRONTEND=noninteractive
# Install build tools and system dependencies for compiling extensions
# libxml2-dev and libxslt-dev are required for lxml compilation
RUN apt-get update && \
    apt-get install -qq -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up conda environment to ensure testbed environment is activated
# First, source conda.sh and activate testbed environment
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate testbed' >> ~/.bashrc

# Set PATH to include conda testbed environment for all users
ENV PATH="/opt/conda/envs/testbed/bin:/opt/conda/bin:$PATH"

# Install pytest in the base environment to ensure it's available regardless of activation
RUN /opt/conda/bin/pip install pytest

# Now switch to testbed environment for all pip installations
RUN bash -lc 'pip install -r requirements.txt'

# Install development dependencies including pytest and other testing tools
RUN bash -lc 'pip install -r dev-requirements.txt'

# Install additional test dependencies from tox.ini (pytest-randomly, coverage, lxml, etc.)
# lxml==4.9.0 requires libxml2-dev and libxslt-dev which are now installed
RUN bash -lc 'pip install pytest-randomly coverage lxml==4.9.0'

# Install optional dependencies that some tests might need
# Based on requirements.txt and README
RUN bash -lc 'pip install \
    ufoLib2==0.14.0 \
    freetype-py==2.3.0 \
    uharfbuzz==0.32.0'

# Install optional extras that might be needed for tests
# Based on tox.ini: ufo, woff, unicode, interpolatable
# We install these before the local package to ensure dependencies are in place
RUN bash -lc 'pip install \
    brotli \
    zopfli \
    fs \
    unicodedata2 \
    scipy'

# Finally, install the project itself in development mode
# This ensures tests will run against the local code
RUN bash -lc 'pip install -e .'

# Verify pytest is accessible from both environments
RUN /opt/conda/bin/pip list | grep pytest && \
    bash -lc 'pip list | grep pytest'

# Also verify that python and pip are accessible
RUN bash -lc 'which python && python --version'
RUN bash -lc 'which pip && pip --version'

# Create a simple script to verify environment activation
RUN echo '#!/bin/bash' > /usr/local/bin/verify-env && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /usr/local/bin/verify-env && \
    echo 'conda activate testbed' >> /usr/local/bin/verify-env && \
    echo 'which python' >> /usr/local/bin/verify-env && \
    echo 'which pytest' >> /usr/local/bin/verify-env && \
    echo 'pytest --version' >> /usr/local/bin/verify-env && \
    chmod +x /usr/local/bin/verify-env