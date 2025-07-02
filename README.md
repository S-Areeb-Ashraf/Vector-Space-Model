# Vector Space Model

This project implements a **Vector Space Information Retrieval System** using **TF-IDF weighting** and **cosine similarity** in Python with NLTK.

## Project Structure

- `Text_files/`: Folder containing the original 448 text documents.
- `Updated_Text_Files/`: Tokenized and stemmed versions of the original files (created by the script).
- `Vector_space_Model.py`: Main script that processes documents, computes TF-IDF vectors, and ranks documents based on cosine similarity to the input query.

## Features

- Tokenization, stop word removal, and stemming using `nltk.PorterStemmer`
- Calculation of:
  - **Term Frequency (TF)**
  - **Inverse Document Frequency (IDF)**
  - **TF-IDF vectors**
- Query processing and ranking using **cosine similarity**
- Filters results based on a similarity threshold

## Requirements

- Python 3.x
- NLTK

Install NLTK using:

```bash
pip install nltk
```

## How It Works

### Preprocessing
1. Reads documents from `Text_files/`
2. Cleans text: removes punctuation, converts to lowercase
3. Removes stop words
4. Applies stemming (PorterStemmer)
5. Saves results to `Updated_Text_Files/`

### Vector Construction
- **TF**: Counts of each word in a document
- **IDF**: `log(total_documents / number_of_documents_with_word)`
- **TF-IDF**: TF × IDF per word per document

### Querying
1. User enters a free-text query.
2. The query is tokenized, stemmed, and converted into a TF-IDF vector.
3. Cosine similarity is calculated between the query vector and all document vectors.
4. Results are filtered and ranked by similarity score.

## How to Run

1. Ensure `Text_files/` folder is present in the same directory as `Vector_space_Model.py`.
2. Open a terminal or command prompt.
3. Run the script:

```bash
python Vector_space_Model.py
```

4. Enter your query when prompted.
5. The system will return a ranked list of document numbers based on similarity.

## Example Query

```
information retrieval model
```

## License

This is a student project for educational purposes. Feel free to modify and build upon it.
