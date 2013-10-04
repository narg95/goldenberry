from setuptools import setup, find_packages

setup(name="Goldenberry",
      packages= find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),    
      entry_points={'orange.widgets': ('Optimization = Goldenberry.widgets.optimization','Learners = Goldenberry.widgets.learners' )},
      package_data = {'': ['icons/*.svg','icons/*.png', '*.ui']},
      zip_safe = False
      )
