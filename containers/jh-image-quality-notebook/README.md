# volumetric Image Quality Assessment Notebook - JupyterHub

[![Build GH Action](https://github.com/3dct/vIQA/actions/workflows/build_and_push_docker_images.yaml/badge.svg)](https://github.com/3dct/vIQA/actions/workflows/build_and_push_docker_images.yaml)
[![latest](https://ghcr-badge.egpl.dev/3dct/image-quality-notebook/latest_tag?label=latest&ignore=dh-*)](https://github.com/3dct/vIQA/pkgs/container/image-quality-notebook)
[![tags](https://ghcr-badge.egpl.dev/3dct/image-quality-notebook/tags?n=3&ignore=dh-*)](https://github.com/3dct/vIQA/pkgs/container/image-quality-notebook)
[![image size](https://ghcr-badge.egpl.dev/3dct/image-quality-notebook/size?tag=jh-latest)](https://github.com/3dct/vIQA/pkgs/container/image-quality-notebook)

This is a single user notebook for volumetric Image Quality Assessment (vIQA) to be used with JupyterHub.

## Base Image
[quay.io/jupyter/base-notebook:python-3.12](https://github.com/jupyter/docker-stacks)

## Additionally Installed Packages
| Package | Tools | JupyterLab Extensions |
|---------|-------|-----------------------|
| viqa    | git   | jupyterlab-git        |
|         |       | jupyter_app_launcher  |
