# Movie Semantic Search (Assignment-1)

This project implements a **semantic search tool** for movies using
Python and SentenceTransformers.\
It allows you to search for movies by meaning rather than exact
keywords.

------------------------------------------------------------------------

## Features

-   Loads movies from `movies.csv` (contains `title` and `plot` columns)
-   Encodes movie plots using `all-MiniLM-L6-v2` embeddings
-   Finds top N movies similar to a query by cosine similarity
-   Clean Jupyter Notebook (`movie_search.ipynb`) with explanations and
    demo
-   Unit tests included to validate functionality
-   GitHub Actions workflow to automatically run tests on every push

------------------------------------------------------------------------

## Project Structure

    Assignment-1/
    ├── movie_search.py           # Core Python implementation
    ├── movie_search.ipynb        # Jupyter notebook version with documentation and demo
    ├── movies.csv                # Dataset of movies with title and plot
    ├── requirements.txt          # Python dependencies
    ├── tests/
    │   └── test_movie_search.py       # Unit tests
    ├── .github/
    │   └── workflows/
    │       └── python-tests.yml       # GitHub Actions CI workflow for automated testing
    ├── .gitignore                # Ignores venv, pycache, and temp files
    └── README.md                 # Project documentation

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/srinidhi151/AI-Systems-Development--IIIT-Naya-Raipur/tree/main/Assignment-1
cd movie-search-assignment
```

### 2. Create a virtual environment (optional but recommended)

``` bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### Run from Python file

``` bash
python movie_search.py
```

------------------------------------------------------------------------

## Testing

To run unit tests:

``` bash
python -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions automatically runs these tests on every push to the
`main` branch.\
You can confirm test results in the **Actions** tab of the repository
(look for a green checkmark).

------------------------------------------------------------------------

## Submission Notes

-   Ensure `.gitignore` excludes `venv/` and `__pycache__/`
-   Verify all files are pushed: code, CSV, tests, notebook, README, and
    workflow
-   Confirm GitHub Actions tests pass before submission
-   Submit the **GitHub repository URL**

------------------------------------------------------------------------

## License

This project is for academic use. You may modify or extend it as needed.
