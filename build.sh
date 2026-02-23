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
