from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='rb_image',
      version=version,
      description="Scikit-image wrapper for Red Buffer. Contains all the utility functions.",
      long_description="""Scikit-image wrapper""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='rb skimage',
      author='imransalam',
      author_email='imran.salam.24@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
