#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import tomlkit
import shutil

project_root = Path(__file__).parent
pyproject_file = project_root / "pyproject.toml"


def update_project_version(new_version):
    try:
        # do backup of the file
        shutil.copy(pyproject_file, str(pyproject_file) + ".bak")

        # Read and parse TOML file
        with open(pyproject_file, "r", encoding="utf-8") as f:
            data = tomlkit.parse(f.read())

        # Update version in the project section
        if "project" in data:
            data["project"]["version"] = new_version
        else:
            print("Error: Could not find project version field in pyproject.toml")
            return False

        # Write updated TOML back to file
        with open(pyproject_file, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(data))

        print(f"Successfully updated version to {new_version}")
        return True

    except Exception as e:
        print(f"Error updating version: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python version.py <new-version>")
        print("Example: python version.py 2.0.0")
        sys.exit(1)

    new_version = sys.argv[1]

    if not os.path.exists(pyproject_file):
        print(f"Error: {pyproject_file} not found in the current directory")
        sys.exit(1)

    update_project_version(new_version)


if __name__ == "__main__":
    main()
