# Movie Semantic Search Assignment

## Objective
This project implements a semantic search system for movies. It uses the `sentence-transformers` library with the `all-MiniLM-L6-v2` model to encode movie plots and return the most relevant movies for a given text query.

---

## Setup
1. Install Python 3.8 or higher.  
2. Create and activate a virtual environment (recommended).  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure `movies.csv` is in the project folder. The file must contain two columns: `title` and `plot`.

---

## Usage
Run the program from the command line:
```bash
python movie_search.py "spy thriller in Paris" --top_n 3
```
The program will output the top matching movies with their similarity scores.

---

## Testing
Unit tests are included in the `tests` folder. Run them with:
```bash
python -m unittest tests/test_movie_search.py -v
```
