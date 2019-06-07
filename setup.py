from setuptools import setup

setup(name='handympi',
      version='0.2',
      description='A simple framework for easy mpi',
      url='http://github.com/sspickle/handympi',
      author='Steve Spicklemire',
      author_email='steve@spvi.com',
      license='MIT',
      packages=['handympi'],
      install_requires=[
          'mpi4py',
      ],
      zip_safe=False)

