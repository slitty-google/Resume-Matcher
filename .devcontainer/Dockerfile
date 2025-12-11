FROM mcr.microsoft.com/devcontainers/javascript-node:20

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Optional: system packages that help with PDF/DOCX processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/resume-matcher
