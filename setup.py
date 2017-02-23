from setuptools import setup, Extension
from io import open
import numpy

requirements = [
	'numpy',
]

setup(
	name = 'networktools',
	description = 'Some measures, surrogates, and other tools for weighted, complete networks.',
	long_description = open('README.rst', encoding='utf8').read(),
	author = 'Gerrit Ansmann, Christian Geier, Kirsten Stahn',
	author_email = 'gansmann@uni-bonn.de',
	url = 'http://github.com/neurophysik/networktools',
	packages = ['networktools'],
	install_requires = requirements,
	classifiers = [
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: BSD License',
		'Operating System :: POSIX',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering :: Mathematics',
		],
	ext_modules = [Extension(
		"networktools._networktools",
		sources = [
			"networktools/_networktools.c",
			"networktools/surrogates.c",
			"networktools/initialchecks.c",
			"networktools/euclid.c",
			"networktools/measures.c",
			],
		extra_compile_args = [
			"-D PYTHON",
			"-fPIC",
			"-Wall",
			"--pedantic",
			"-std=c11",
			"-Wno-unknown-pragmas",
			"-Wno-incompatible-pointer-types-discards-qualifiers",
			"-Wno-unused-function"
			],
		extra_link_args = [ "-lm", "-lgsl", "-lgslcblas" ],
		include_dirs = [numpy.get_include()]
		)],
	verbose = True
)

