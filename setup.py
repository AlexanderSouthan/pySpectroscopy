from setuptools import setup, find_packages

setup(
    name='pyAnalytics',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'matplotlib', 'pandas', 'tqdm', 'imageio', 'scikit-learn', 'scipy', 'pip', 'plotly', 'little_helpers', 'pyPreprocessing', 'pyDataFitting', 'spc']
)
