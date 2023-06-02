from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="SLML",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2.0.0',
    packages=find_packages(include=['SLML', 'SLML.*']),
    description="Statistical Learning Models Library",
    install_requires=[
      'numpy>=1.22.4',
      'OMADS>=1.5.0',
      'pandas>=1.4.2',
      'pyDOE2>=1.3.0',
      'scikit_learn>=1.1.1',
      'scipy>=1.8.1',
      'setuptools>=58.1.0',
      'requests>=2.20.0'
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
  )