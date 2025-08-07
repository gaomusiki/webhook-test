#!/bin/bash

# from light ubuntu image or ngc pytorch image
build_light_base=false # only build the base light image without `ref`
from_light_image=true

docker_prefix=docker.io/library
docker_file="Dockerfile"
docker_name="a4_env"

if [[ $build_light_base = true ]]; then
    docker_file="Dockerfile.light_base"
    docker_name="a_env_light_base"
else
    if [[ $from_light_image = true ]]; then
        docker_file="${docker_file}.light"
        docker_name="${docker_name}_light"
    fi
fi


docker_version=2

# build docker image
docker build -f "$docker_file" -t "$docker_name:v$docker_version" .

# save docker image tar
docker save -o "${docker_name}_v${docker_version}".tar "$docker_prefix/$docker_name:v$docker_version"