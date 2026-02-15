from setuptools import setup, find_packages

setup(
  name="tinytensor",
  version="0.2.0",
  packages=find_packages(),
  include_package_data=True,
  package_data={
    "tinytensor": [
      "engine/**/*.so",
    ]
  },
  python_requires=">=3.8",
  author="Sahil Rajwar",
  url="https://github.com/0xhilSa/tinytensor",
  license="MIT",
  classifiers=[
    "License :: OSI Approved :: MIT License",
  ],
  description="A lightweight tensor computation library",
)
