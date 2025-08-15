
from setuptools import setup, find_packages

setup(
    name='scene_grounding',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers',
        'numpy',
        'opencv-python',
        'Pillow',
        'matplotlib',
        'scipy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            # Define entry points if any scripts are available to run
        ],
    },
    author='Vansh Whig',
    author_email='vansh.whig@example.com',
    description='Scene Grounding Model Implementation',
    url='https://github.com/codemaster-vansh/Scene_Grounding'
)
