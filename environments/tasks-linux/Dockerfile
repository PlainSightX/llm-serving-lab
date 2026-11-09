FROM python@sha256:cd7c412d000912f29075a1b8803e43cb2f38bb67f104019df526df5ceaf30569
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 UV_LINK_MODE=copy CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install --no-cache-dir uv==0.12.5 cmake==3.31.6
WORKDIR /opt/q1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --python /usr/local/bin/python
ENV PATH=/opt/q1/.venv/bin:$PATH PYTHONPATH=/code
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-u"]
