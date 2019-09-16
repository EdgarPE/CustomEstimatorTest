#!/usr/bin/env sh

docker run \
    -it \
    --rm \
    --name estimator-console \
    --entrypoint /bin/bash \
    estimator-app