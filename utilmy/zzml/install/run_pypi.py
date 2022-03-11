# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Pypi Uploader

python pypi.py


1) Create the package in one folder


2)
A file called .pypirc that has to contain login credentials.
  $ ~ YOURTEXTEDITOR ~/.pypirc
  Open a file and paste this to in it:

  [pypi]
  username = token
  password = pypi-AgEI


"""


import os
import os.path as op
import re
import subprocess
import sys

curdir = op.abspath(op.curdir)
setup_file = op.join(curdir, 'setup.py')




class Version(object):
    pattern = re.compile(r"(version\s*=\s*['\"]\s*(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)\s*['\"])")

    def __init__(self, major, minor, patch):
        """ Version:__init__
        Args:
            major:     
            minor:     
            patch:     
        Returns:
           
        """
        self.major = int(major)
        self.minor = int(minor)
        self.patch = int(patch)

    def __str__(self):
        """ Version:__str__
        Args:
        Returns:
           
        """
        return f'Version({self.stringify()})'

    def __repr__(self):
        """ Version:__repr__
        Args:
        Returns:
           
        """
        return self.__str__()

    def stringify(self):
        """ Version:stringify
        Args:
        Returns:
           
        """
        return f'\'{self.major}.{self.minor}.{self.patch}\''

    def new_version(self, orig):
        """ Version:new_version
        Args:
            orig:     
        Returns:
           
        """
        return '='.join([orig.split('=')[0], self.stringify()])

    @classmethod
    def parse(cls, string):
        """ Version:parse
        Args:
            cls:     
            string:     
        Returns:
           
        """
        re_result = re.findall(cls.pattern, string)
        if len(re_result) == 0:
            return Exception('Program was not able to parse version string, please check your setup.py file.')

        return re_result[0][0], cls(*re_result[0][1:])


def ask(question, ans='yes'):
    """function ask
    Args:
        question:   
        ans:   
    Returns:
        
    """
    return input(question).lower() == ans.lower()

def pypi_upload():
    """
      It requires credential in .pypirc  files
      __token__

    """
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    print("Upload files")

    print("Deleting build files")
    for item in os.listdir(op.abspath(op.join(setup_file, '..'))):
        if item.endswith('.egg-info') or item in ['dist', 'build']:
            os.system(f'rm -rf {item}')



def update_version(path, n):
    """function update_version
    Args:
        path:   
        n:   
    Returns:
        
    """
    content = open(path, 'r').read()
    
    orig, version = Version.parse(content)
    print (f'Current version: {version}')

    version.minor += int(n)
    print (f'New Version: {version}')

    with open(path, 'w') as file:
        file.write(content.replace(orig, version.new_version(orig)))


def git_commit(message):
    """function git_commit
    Args:
        message:   
    Returns:
        
    """
    if not ask(f'About to git commit {message}, are you sure: '):
        exit()

    os.system(f'git commit -am "{message}"')
    
    if not ask('About to git push, are you sure: '):
        exit()

    os.system('git push')


def main(*args):
    """function main
    Args:
        *args:   
    Returns:
        
    """
    print ('Program Started')
    update_version(setup_file, 1)
    # git_commit(*sys.argv[1:3])
    pypi_upload()
    print ('Upload success')



if __name__ == '__main__':
    main()

    #if len(sys.argv) == 1:
    #    print (f'Usage: python {sys.argv[0]} "commmit message"'); sys.exit()
    #
    #main(*sys.argv[1:3])
