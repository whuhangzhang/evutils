from setuptools import find_packages, setup

install_requires = [
    line.strip() for line in open('requirements.txt', 'r').readlines()
]


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'evutils/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='evutils',
    version=get_version(),
    description='EV-Utils is a personal python toolbox.',
    long_description=readme(),
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities',
    ],
    url='https://github.com/whuhangzhang/evutils',
    author='Hang Zhang',
    author_email='whuhangzhang@gmail.com',
    license='GPLv3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    zip_safe=False)
