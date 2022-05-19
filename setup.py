import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertpatentclass",
    version="0.1.0",
    author="higashi-masaki",
    author_email="ls16287j@gmail.com",
    description="You can predict class from BERT with patent text",
    url="https://github.com/M-H0127/patent_clas",
    packages=setuptools.find_packages(),
    #install_requires=["numpy", "scikit-learn", "torch", "transformers", "tqdm", "unidic-lite", "unidic", "fugashi", "ipadic"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)