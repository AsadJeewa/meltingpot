#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs meltingpot on Linux/macOS.

set -euxo pipefail


function check_version_gt() {
  local required="$1"
  local input lowest
  input="$(grep -Eo '[0-9]+\.[0-9]+' /dev/stdin | head -n 1)"
  lowest="$(printf "${required}\n${input}" | sort -V | head -n 1)"
  [[ "${lowest}" == "${required}" ]]
}


function check_setup() {
  echo -e "\nChecking python version..."
  python --version
  python --version | check_version_gt '3.9'
}


function install_meltingpot() {
  echo -e "\nInstalling meltingpot..."
  pip install --upgrade pip build setuptools
  python -m build --sdist --outdir dist/
  pip install dist/*.tar.gz
}


function main() {
  check_setup
  install_meltingpot
}


main "$@"
