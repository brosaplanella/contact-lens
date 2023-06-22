# Contact lens

> **Warning**
> The code in this repository is work in progress. 

Code to solve the models for contact lens polymerisation from the Johnson & Johnson problem in the ESGI 173 Limerick.

## How to install?
These installation instructions assume you have Python installed (versions 3.9 to 3.11) and that you have also installed the `virtualenv` package which can be done by running
```bash
pip install virtualenv
```
You should first clone the repository, see [instructions on how to do it](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), and get into the folder.

### Linux & MacOS
1. Create a virtual environment (this is strongly recommended to avoid clashes with the dependencies).
```bash
virtualenv env
```

2. Activate the virtual environment
```bash
source env/bin/activate
```
The virtual environment can later be deactivated (if needed) by running
```bash
deactivate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

### Windows
1. Create a virtual environment (this is strongly recommended to avoid clashes with the dependencies).
```bash
python -m virtualenv env
```

2. Activate the virtual environment
```bash
env\Scripts\activate.bat
```
The virtual environment can later be deactivated (if needed) by running
```bash
deactivate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

## How to use the code?