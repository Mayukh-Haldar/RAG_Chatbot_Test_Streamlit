# setup.py
from setuptools import find_packages, setup
from typing import List
import os


def get_requirements(filepath: str) -> List[str]:
    requirements = []
    # Use 'utf-8' encoding for broader compatibility
    with open(filepath, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            # Ignore comments, empty lines, and editable installs like '-e .'
            if line and not line.startswith('#') and not line.startswith('-e'):
                requirements.append(line)
    return requirements


setup(
    name="rag-chatbot-streamlit",  # New name maybe?
    version="0.1.0",
    description="A Streamlit RAG Chatbot using Nomic and Groq.",
    author="Mayukh Haldar",
    author_email="mayukhhaldar1@gmail.com",
    packages=find_packages(where='rag_app'),  # Find packages in rag_app
    package_dir={'': 'rag_app'},  # Map root namespace to rag_app dir
    include_package_data=True,
    install_requires=get_requirements(
        "requirements.txt"),  # Reads root requirements
    entry_points={},  # No console scripts needed for Streamlit Cloud deploy
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
