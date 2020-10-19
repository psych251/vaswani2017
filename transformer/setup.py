from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='transformer',
      packages=find_packages(),
      version="0.1.0",
      description='A reproducibility project of vaswani et al 2017',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/transformer.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm",
                         "psutil"],
      py_modules=['transformer'],
      long_description='''
            A reproducibility project of vaswani et al 2017
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
