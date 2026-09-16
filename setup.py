from setuptools import find_packages,setup
HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str):
    """This function returns a list of requirements"""
    requirements=[]
    with open(file_path) as file:
        content=file.readlines()
        requirements=[req.strip() for req in content]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name="mlProject",
    version='0.0.1',
    author= "Sambhav",
    author_email= "sambha.8761@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)