import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='quax',
        version="0.2.0a1",
        description='Arbitrary order derivatives of electronic structure computations.',
        author='Adam Abbott, Erica Mitchell',
        author_email='adabbott@uga.edu, emitchell@uga.edu',
        url="none",
        license='BSD-3C',
        packages=setuptools.find_packages(where="quax"),
        package_dir={"": "quax"},
        install_requires=[
            'numpy>=1.23',
            'jax>=0.4.19',
            'jaxlib>=0.4.19',
            'h5py>=2.8.0',
            'scipy>=1.9'
        ],
        extras_require={
            'tests': [
                'pytest-cov',
            ],
        },

        tests_require=[
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False
    )
