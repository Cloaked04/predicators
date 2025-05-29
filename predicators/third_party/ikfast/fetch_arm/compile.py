import fnmatch
import importlib
import os
import shutil
from distutils.core import setup
from distutils.dir_util import copy_tree
from distutils.extension import Extension

# Build C++ extension by running: 'python setup.py build'
# see: https://docs.python.org/3/extending/building.html

# http://openrave.org/docs/0.8.2/openravepy/ikfast/
# https://github.com/rdiankov/openrave/blob/master/python/ikfast.py#L92
# http://wiki.ros.org/Industrial/Tutorials/Create_a_Fast_IK_Solution

# Yijiang
# https://github.com/yijiangh/ikfast_pybind
# https://github.com/yijiangh/conrob_pybullet/tree/master/utils/ikfast
# https://github.com/yijiangh/choreo/blob/bc777069b8eb7283c74af26e5461532aec3d9e8a/framefab_robot/abb/framefab_irb6600/framefab_irb6600_support/doc/ikfast_tutorial.rst


# def compile_ikfast(module_name, cpp_filename, remove_build=False):
#     ikfast_module = Extension(module_name, sources=[cpp_filename])
#     setup(name=module_name,
#           version='1.0',
#           description="ikfast module {}".format(module_name),
#           ext_modules=[ikfast_module])

#     build_lib_path = None
#     for root, dirnames, filenames in os.walk(os.getcwd()):
#         if fnmatch.fnmatch(root, os.path.join(os.getcwd(), "*build", "lib*")):
#             build_lib_path = root
#             break
#     assert build_lib_path

#     copy_tree(build_lib_path, os.getcwd())
#     if remove_build:
#         # TODO: error when compiling multiple arms for python2
#         # error: unable to open output file 'build/temp.macosx-10.15-x86_64-2.7/movo_right_arm_ik.o': 'No such file or directory'
#         shutil.rmtree(os.path.join(os.getcwd(), 'build'))

#     try:
#         importlib.import_module(module_name)
#         print('\nikfast module {} imported successful'.format(module_name))
#     except ImportError as e:
#         print('\nikfast module {} imported failed'.format(module_name))
#         raise e
#     return True


def compile_ikfast(module_name, cpp_filename, remove_build=False):
    print(f"\nCompiling IKFast module: {module_name}")
    print(f"Using source file: {cpp_filename}")
    
    ikfast_module = Extension(
        module_name, 
        sources=[cpp_filename],
        extra_compile_args=['-v']  # Add verbose flag to compiler
    )
    
    setup(name=module_name,
          version='1.0',
          description="ikfast module {}".format(module_name),
          ext_modules=[ikfast_module],
          verbose=3)  # Maximum verbosity for setup
    
    print("\nSearching for build output...")
    build_lib_path = None
    for root, dirnames, filenames in os.walk(os.getcwd()):
        if fnmatch.fnmatch(root, os.path.join(os.getcwd(), "*build", "lib*")):
            build_lib_path = root
            print(f"Found build path: {build_lib_path}")
            break
    
    assert build_lib_path, "Build path not found!"
    
    print(f"Copying from {build_lib_path} to {os.getcwd()}")
    copy_tree(build_lib_path, os.getcwd())
    
    if remove_build:
        print(f"Removing build directory")
        shutil.rmtree(os.path.join(os.getcwd(), 'build'))
    
    try:
        importlib.import_module(module_name)
        print(f'\nIKFast module {module_name} imported successfully')
    except ImportError as e:
        print(f'\nIKFast module {module_name} import failed')
        raise e
    
    return True
