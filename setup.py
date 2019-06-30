
import os
import subprocess
import multiprocessing

from setuptools import setup
from distutils.command import build as build_module

base_path = os.path.dirname(__file__)

with open(os.path.join(base_path, 'README.md')) as f:
    long_description = '\n' + f.read()

def build_csrc():
    build_path = os.path.join(base_path, 'build')
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    subprocess.check_call(['cmake', os.path.join(base_path, 'csrc')], cwd=build_path)
    build_cpus = min(multiprocessing.cpu_count(), 4)
    subprocess.check_call(['make', '-j{}'.format(build_cpus)], cwd=build_path)

class CmakeAndBuild(build_module.build):
    def run(self):
        build_csrc()
        build_module.build.run(self)

setup(
    name='dirt',
    version='0.3.0',
    description='DIRT: Differentiable Renderer for TensorFlow',
    long_description=long_description,
    author='Paul Henderson',
    author_email='paul@pmh47.net',
    url='https://github.com/pmh47/dirt',
    packages=['dirt'],
    python_requires='>=2.7.0',
    install_requires=['tensorflow-gpu>=1.6', 'numpy'],
    package_data={'dirt': ['*.so', '*.dll']},
    include_package_data=True,
    cmdclass={'build': CmakeAndBuild},
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
