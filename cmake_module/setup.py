from setuptools import setup, Extension
import sys
import os
import subprocess
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        
class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                 '-DPYTHON_EXECUTABLE=' + sys.executable
            ]

            build_args = ['--config', 'Release']

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            # Build
            subprocess.check_call(['cmake', '--build', '.'] + build_args,
                                  cwd=self.build_temp)

#class my_build_ext(build_ext):
#    def build_extensions(self):
#        if self.compiler.compiler_type == 'msvc':
#            raise Exception("Visual Studio is not supported")
#
#        for e in self.extensions:
#            e.extra_compile_args = ['--pedantic']
#            e.extra_link_args = ['-lgomp']
#
#        build_ext.build_extensions(self)

setup(name = "mymath",
      version = "0.1",
      ext_modules = [CMakeExtension("mymath")],
      #cmdclass = {'build_ext': my_build_ext},
      cmdclass=dict(build_ext=cmake_build_ext),
      );
