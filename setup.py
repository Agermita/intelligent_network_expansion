from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='traficforcast',
      version="0.0.01",
      description="traficforcast Model (automate_model_lifecycle)",
      license="MIT",
      author="Le Wagon batch 1173",
      author_email="aouatif.agermit@orange.com",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
