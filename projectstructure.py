import os
from pathlib import Path


while True:
    project_name = input("Enter the Source Folder Name: ")
    if project_name != '':
        break


list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/app/__init__.py",
    f"{project_name}/app/api_utils.py",
    f"{project_name}/app/chat_interface.py",
    f"{project_name}/app/sidebar.py",
    f"{project_name}/app/streamlit_app.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/exception.py",
    f"{project_name}/pipeline/logger.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/chroma_utils.py",
    f"{project_name}/utils/db_utils.py",
    f"{project_name}/utils/langchain_utils.py",
    f"{project_name}/utils/main.py",
    f"{project_name}/utils/pydantic_models.py",
    "notebook/research.ipynb",
    "init_setup.sh",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
