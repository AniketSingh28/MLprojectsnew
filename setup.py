# from setuptools import find_packages,setup
# from typing import List
# Hypen_e_dot='-e.'
# def get_requirements(file_path:str)-> List[str]:
#     '''
#     This function will return the LIST of the requirements
#     '''
#     requirments=[]
#     with open (file_path) as file_obj:
#         requirements=file_obj.readlines()
#         requirements=[req.replace("\n","") for req in requirements]
#         if Hypen_e_dot in requirements:
#             requirements.remove(Hypen_e_dot)
#     return requirements 


# setup(
# name='ml project',
# version='0.0.1',
# author='Aniket Singh',
# author_email='aniketshanjul@gmail.com'
# packages=find_packages(),
# install_requires=get_requirements['requirements.txt']
# )


from setuptools import find_packages, setup
from typing import List

Hyphen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of the requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if Hyphen_e_dot in requirements:
            requirements.remove(Hyphen_e_dot)
    return requirements

setup(
    name='ml project',
    version='0.0.1',
    author='Aniket Singh',
    author_email='aniketshanjul@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
