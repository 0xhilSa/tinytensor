import os
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


ext_modules = [
  Extension("tinytensor.engine.cpu.cpu", sources=[]),
  Extension("tinytensor.engine.cpu.cpu_ops", sources=[]),
  Extension("tinytensor.engine.cpu.functional_cpu", sources=[]),
  Extension("tinytensor.engine.constants", sources=[]),
  Extension("tinytensor.engine.cuda.cuda", sources=[]),
  Extension("tinytensor.engine.cuda.cuda_ops", sources=[]),
  Extension("tinytensor.engine.cuda.functional_cuda", sources=[]),
]


class BuildNVCC(build_ext):
  def build_extension(self, ext):
    py_inc = subprocess.check_output(["python3", "-c", "import sysconfig; print(sysconfig.get_paths()['include'])"]).decode().strip()

    GENCODE = (
      "-gencode arch=compute_86,code=sm_86 "
      "-gencode arch=compute_86,code=compute_86 "
    )

    TEN_SRC = "tinytensor/engine"
    C_SRC_DIR = "tinytensor/engine/cpu"
    CU_SRC_DIR = "tinytensor/engine/cuda"

    output_path = self.get_ext_fullpath(ext.name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if ext.name.endswith("cpu"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{C_SRC_DIR} "
        f"{C_SRC_DIR}/cpu.c {TEN_SRC}/tensor.c "
        f"-lcudart -o {output_path}"
      )
    elif ext.name.endswith("cpu_ops"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{C_SRC_DIR} "
        f"{C_SRC_DIR}/cpu_ops.c {TEN_SRC}/tensor.c "
        f"-lcudart -o {output_path}"
      )
    elif ext.name.endswith("functional_cpu"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{C_SRC_DIR} "
        f"{C_SRC_DIR}/functional_cpu.c {TEN_SRC}/tensor.c "
        f"-o {output_path}"
      )
    elif ext.name.endswith("constants"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} "
        f"{TEN_SRC}/constants.c "
        f"-o {output_path}"
      )
    elif ext.name.endswith("cuda") and not ext.name.endswith("functional_cuda"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{CU_SRC_DIR} "
        f"{CU_SRC_DIR}/cuda.cu {TEN_SRC}/tensor.c "
        f"-lnvidia-ml -o {output_path}"
      )
    elif ext.name.endswith("cuda_ops"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{CU_SRC_DIR} "
        f"{CU_SRC_DIR}/cuda_ops.cu {TEN_SRC}/tensor.c "
        f"-o {output_path}"
      )
    elif ext.name.endswith("functional_cuda"):
      cmd = (
        f"nvcc -O3 {GENCODE} -Xcompiler -fPIC -shared "
        f"-I{py_inc} -I{CU_SRC_DIR} "
        f"{CU_SRC_DIR}/functional_cuda.cu {TEN_SRC}/tensor.c "
        f"-o {output_path}"
      )
    else:
        raise RuntimeError(f"Unknown extension: {ext.name}")

    print(f"Compiling: {cmd}")
    subprocess.check_call(cmd, shell=True)


readme_path = "README.md"
long_description = open(readme_path).read() if os.path.exists(readme_path) else ""

setup(
  name="tinytensor",
  version="0.3.0",
  author="Sahil Rajwar",
  description="A lightweight tensor computation library",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  include_package_data=True,
  ext_modules=ext_modules,
  cmdclass={"build_ext": BuildNVCC},
  url="https://github.com/0xhilSa/tinytensor",
  license="MIT",
  license_files=("LICENSE",),
  classifiers=[
    "Programming Language :: Python :: 3",
    "Programming Language :: C",
    "Programming Language :: Python :: Implementation :: CPython",
    "Operating System :: POSIX :: Linux",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ],
  python_requires=">=3.9",
)
