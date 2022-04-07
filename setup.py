from distutils.core import setup

setup(name='SWAflow',
      version='0.1',
      packages=['swaflow'],
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      description='Tensorflow implementation of stochastic weighting averaging',
      long_description='Tensorflow implementation of stochastic weighting averaging',
      entry_points = {'console_scripts': [
          'swaflow=swaflow.command_line:main']},
      author='Pablo Lemos',
      author_email='p.lemos@sussex.ac.uk',
      url='https://github.com/Pablo-Lemos/SWAflow'
      )