#! /usr/bin/env python
""" The setup file for the fiteos project.
"""

import os
import subprocess
from distutils import core
try:
    from setuptools.command import install as _install
except:
    from distutils.command import install as _install

def find_package_data(dirname):
    """ Returns a list of all relative file paths under ``dirname`` for files
    that do not end in ``.py`` or ``.pyc``.

    Parameters
    ----------
    dirname : str
        Path of directory to check.

    Returns
    -------
    list
        A ``list`` of relative file paths.
    """
    def find_paths(dirname):
        items = []
        for fname in os.listdir(dirname):
            path = os.path.join(dirname, fname)
            if os.path.isdir(path):
                items += find_paths(path)
            elif not path.endswith((".py", ".pyc")):
                items.append(path)
        return items
    items = find_paths(dirname)
    return [os.path.relpath(path, dirname) for path in items]

# call run function from setup
def do_setup(*args):
    return True
_install.install._called_from_setup = do_setup

# meta-data about the package
project_name = "fiteos"
project_version = "0.0.dev0"
project_url = "https://gitlab.lanl.gov/cmbiwer/fiteos"
project_description = "Equation of state fitting"
project_keywords = ["EOS", "equation of state"]

# a list of required packages to run project
install_requires = [
]

# a list of all executables to be installed
scripts_list = [
    "bin/fiteos_fit",
]

# a list of all python packages to be installed
packages_list = [
    "fiteos",
    "fiteos.equations",
    "fiteos.io",
    "fiteos.plots"
]

# a dict of all data to be installed
data_dict = {
    "fiteos" : find_package_data("fiteos"),
}

# a dict for commands
cmd_dict = {
}

# run setup
core.setup(name=project_name,
           version=project_version,
           description=project_description,
           url=project_url,
           keywords=project_keywords,
           install_requires=install_requires,
           scripts=scripts_list,
           packages=packages_list,
           package_data=data_dict,
           cmdclass=cmd_dict,
           zip_safe=False)