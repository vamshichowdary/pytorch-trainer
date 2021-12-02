import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='pytorch-trainer-tensorboard',  
    version='0.1',
    license='MIT',
    author="Vamshi Chowdary Madala",
    author_email="vamchowdary72@gmail.com",
    description="A simple trainer class for pytorch with tensorboard support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vamshichowdary/pytorch-trainer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[]                     # Install other dependencies if any
)