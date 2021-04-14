import os
import setuptools
from goodDataML import __version__

here = os.path.abspath(os.path.dirname(__file__))


def read_file(file_name):
    file_path = os.path.join(here, file_name)
    return open(file_path).read().strip()


setuptools.setup(
    name="goodDataML",  # Replace with your own username
    version=__version__,
    author="dev-goodata",
    author_email="dev@goodata.org",
    description="ML SDK",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests*', 'examples', 'run']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Software Development :: Build Tools',
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)