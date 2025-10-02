#!/bin/bash

set -eu

curl -sOf https://cdn.intra.42.fr/document/document/39824/leaves.zip
unzip leaves.zip
pushd images
    mkdir -vp Apples Grapes
    mv Apple_* Apples/
    mv Grape_* Grapes/
popd
rm leaves.zip