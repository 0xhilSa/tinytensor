#!/usr/bin/bash

VERSION="0.3.0"
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknow")
CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
VERSION="${VERSION}+cu${CUDA_VERSION//./}"

cat > tinytensor/version.py << EOF
# auto generted
from typing import Optional

__all__ = ["__version__", "cuda", "git_version"]
__version__ = "${VERSION}"
cuda: Optional[str] = "${CUDA_VERSION}"
git_version = "${GIT_HASH}"
EOF


pip install -r ./requirements.txt
python3 -m build --wheel
pip install dist/*.whl --force-reinstall
rm -rf ./*.egg-info/ ./dist ./build
