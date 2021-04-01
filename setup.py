from setuptools import setup, find_packages
setup(
    name='myutil',
    version='0.0.3',
    packages=find_packages(),
    entry_points={
        'console_scripts':
            [


            ],
    },
    license='MIT',
    description='A simple commandline utility for python scripts.',
    keywords=['PYTHON', 'CLI', 'UTILITIES'],
    install_requires=[
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
)

