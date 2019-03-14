from setuptools import setup


version = '0.1'

setup(name='rb_image',
      version=version,
      description="Scikit-image wrapper for Red Buffer. Contains all the utility functions.",
      long_description=open('README.md').read(),
      author='Imran us Salam, Mian',
      author_email='imran.salam.24@gmail.com',
      license='MIT',
      packages=['rb_image'],
      include_package_data=True,
      install_requires=['scikit-image'],
      )
