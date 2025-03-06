FROM quay.io/jupyter/base-notebook:python-3.12
LABEL org.opencontainers.image.title="volumetric Image Quality Assessment Notebook - JupyterHub"
LABEL org.opencontainers.image.description="This is a custom image for volumetric image quality assessment in jupyterhub."
LABEL org.opencontainers.image.authors="Lukas Behammer, lukas.behammer@fh-wels.at"
LABEL org.opencontainers.image.source="https://github.com/3dct/vIQA/tree/main/containers/jh-image-quality-notebook/Dockerfile"
LABEL org.opencontainers.image.documentation="https://github.com/3dct/vIQA/tree/main/containers/jh-image-quality-notebook/README.md"
LABEL org.opencontainers.image.url="https://github.com/3dct/vIQA"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

# install git
USER root
RUN apt-get update
RUN apt-get install -y git-all

USER ${NB_UID}
# install additional packages
RUN pip install viqa
RUN pip install jupyter_app_launcher
RUN pip install jupyterlab-git

COPY notebooks /usr/share/jupyter/jupyter_app_launcher/notebooks
COPY containers/config/jp_app_launcher/jp_app_launcher.yaml /usr/share/jupyter/jupyter_app_launcher/jp_app_launcher.yaml
