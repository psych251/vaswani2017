from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='crab',
      packages=find_packages(),
      version="0.1.0",
      description='A visual attention system',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/voduct.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm",
                         "psutil"],
      py_modules=['crab'],
      long_description='''
          This is a package to create and use a combination method of
          transformers with CNNs leverage attention for improved
          compositional feature finding and counting of objects in
          static images.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
