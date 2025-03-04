from setuptools import setup, find_packages
# To install: run pip install  --config-settings editable_mode=compat .
setup(
    name='SEDer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        # e.g., 'numpy', 'pandas', 'scipy'
    ],
    author='Ironi & Shahaf',
    author_email='your.email@example.com',
    description='A brief description of your package',
    url='https://github.com/yourusername/SED-WIS',  # Update with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)