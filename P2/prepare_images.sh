#!/bin/bash

set -eu

curl -sOf https://cdn.intra.42.fr/document/document/39824/leaves.zip
unzip leaves.zip
rm leaves.zip
