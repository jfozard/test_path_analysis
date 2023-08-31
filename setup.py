from setuptools import setup, find_packages

setup(
    name='path_analysis',
    version='0.1.0',
    description='A brief description of your package',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/yourrepository',  # if you have a repo for the project
    packages=find_packages(),  # or specify manually: ['your_package', 'your_package.submodule', ...]
    install_requires=[
        'numpy',  # for example, if your package needs numpy
        'gradio',
        # ... other dependencies
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # ... other classifiers
    ],
    python_requires='>=3.6',  # your project's Python version requirement
    keywords='some keywords related to your project',
    # ... other parameters
)
