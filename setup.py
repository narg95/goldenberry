from setuptools import setup, find_packages

NAME = 'Goldenberry'
VERSION = '1.0'
DESCRIPTION = 'A supplementary Machine Learning and Evolutionary Computation suite for Orange.'
AUTHOR = 'Nestor Andres Rodriguez, Sergio G. Rojas, District University FJC, Bogota, Colombia.'
AUTHOR_EMAIL = 'nearodriguezg@correo.udistrital.edu.co, srojas@udistrital.edu.co'
URL = 'http://Goldenberry.codeplex.com'
DOWNLOAD_URL = 'https://Goldenberry.codeplex.com/releases'
LICENSE = 'GPLv2'
KEYWORDS = (
'data mining',
'machine learning',
'artificial intelligence',
'optimization',
'evolutionary computation',
'edas',
'wkiera',
'perceptron',
'feature selection',
'orange',
'orange add-on',
)
PACKAGES = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
PACKAGE_DATA =  {'': ['icons/*.svg','icons/*.png', '*.ui']}
SETUP_REQUIRES = ('setuptools',)
INSTALL_REQUIRES = ('Orange',
                    'setuptools',
                    'numpy',
                    'scipy')
ENTRY_POINTS = {'orange.addons': ('Goldenberry = Goldenberry'),
                'orange.widgets': ('Optimization = Goldenberry.widgets.optimization','Learners = Goldenberry.widgets.learners' )}

if __name__ == '__main__':
    setup(
            name = NAME,
            version = VERSION,
            description = DESCRIPTION,
            author = AUTHOR,
            author_email = AUTHOR_EMAIL,
            url = URL,
            download_url = DOWNLOAD_URL,
            license = LICENSE,
            keywords = KEYWORDS,
            packages = PACKAGES,
            package_data = PACKAGE_DATA,
            setup_requires = SETUP_REQUIRES,
            install_requires = INSTALL_REQUIRES,
            entry_points = ENTRY_POINTS,
            include_package_data = True,
            zip_safe = False,)
