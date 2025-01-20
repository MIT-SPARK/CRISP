from setuptools import setup, find_packages

setup(
    # continuous registration of object pose and shape
    name='crisp',
    version='0.1.0',
    author='Jingnan Shi',
    author_email='jnshi@mit.edu',
    python_requires='>=3.9',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        # no requirements for easy debugging
    ],
    entry_points={
        'console_scripts': [
            # potentially add training scripts here
        ],
    },
)