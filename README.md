## Overview

This README provides information on using the document retrieval system implemented in `IR_engine.py` and `my_retriever.py`. The system processes queries against a pre-built inverted index, computes similarity scores using various weighting schemes, and outputs the ranked results. For more details about the project, look at [report](report/Information_Retrieval_Report.pdf).

## Usage

To run the document retrieval system, use the following command line syntax:

```sh
python IR_engine.py [options]
```

## Options

    -h : Print this help message.
    -s : Use the "with stoplist" configuration (default: without stoplist).
    -p : Use the "with stemming" configuration (default: without stemming).
    -w LABEL : Use weighting scheme "LABEL" (LABEL must be one of: binary, tf, tfidf; default: binary).
    -o FILE : Output results to file FILE (this option is mandatory).

## Example

To run the document retrieval system with TF-IDF weighting, using both stoplist and stemming, and output the results to output.txt:

```sh
python IR_engine.py -s -p -w tfidf -o output.txt
```

## Implementation Details
### Files

    IR_engine.py: Main script to run the document retrieval system.
    my_retriever.py: Contains the Retrieve class which handles the query processing and document retrieval.
    IR_data.pickle: Pre-built data containing the inverted index and queries.

## Dependencies

    Python 3.x
    pickle for loading pre-built data.
    sys and getopt for command line processing.
