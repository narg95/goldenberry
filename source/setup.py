from setuptools import setup, find_packages

setup(name="Goldenberry",
      packages= ["Goldenberry", "Goldenberry.widgets"],      
      entry_points={"orange.widgets": ("Goldenberry = Goldenberry.widgets.optimization")},
      package_data = {'': ['icons/*.svg']},
      zip_safe = False,
      )
