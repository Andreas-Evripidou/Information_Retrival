# Author: Andreas Evripidou
# Date: 06/11/2023
# Description: This Python script implements a document retrieval system using the Vector Space Model.
import math


class Retrieve:
    #==============================================================================
    # Init

    # Create new Retrieve object storing index and term weighting 
    # scheme. ​(You can extend this method, as required.)
    def __init__(self,index,term_weighting):
        # Store the index and term weighting scheme
        self.index = index
        self.term_weighting = term_weighting
        # Compute the number of documents in the collection
        self.num_docs = self.compute_number_of_documents()
        # Compute the document term matrix
        self.document_vectors_length = self.compute_document_vector_lengths()
        
    def compute_number_of_documents(self): 
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method for computing the length of each document vector 
    # returns a dictionary of document ids and their length
    # document_vector_lengths = {doc_id: length}
    def compute_document_vector_lengths(self):
        # Initialise the document vectors length
        document_vector_lengths = {}

        # For each term in the index
        for term in self.index.keys():
            # For each document that contains the term
            for doc in self.index[term]:
                # If the document is already in the document vectors length
                # dictionary, add the term weight squared to the length
                # Otherwise, add the document to the dictionary and set the
                # length to the term weight squared
                if doc in document_vector_lengths:
                    document_vector_lengths[doc] += self.get_weighting(term, doc) ** 2
                else:
                    document_vector_lengths[doc] = self.get_weighting(term, doc) ** 2
        
        # For each document in the document vectors length dictionary
        # compute the square root of the length
        document_vector_lengths = {
            doc: math.sqrt(document_vector_lengths[doc]) 
            for doc in document_vector_lengths.keys() }
        
        return document_vector_lengths
    
    #==============================================================================
    # Query Processing

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). ​Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):

        # Filter query to only include known terms
        query = [term for term in query if term in self.index.keys()]

        # Filter the documents to only include the ones that contain 
        # at least one of the query terms
        relevant_docs = set()
        relevant_docs = [doc for term in query for doc in self.index[term]] 
        
        # Compute the query vector and the document vectors
        query_vector, document_vectors = self.compute_vectors(query)
        
        # Compute the cosine similarity between the query and each document
        scores_dict = {}
        scores_dict = {
            doc_id: self.cosine_similarity(query_vector, document_vectors[doc_id], doc_id)
            for doc_id in relevant_docs
            }

        # Rank the documents based on the cosine similarity
        ranked_docs = self.rank_documents(scores_dict)
        return ranked_docs[:10]
    
    # Method ranking the documents based on the cosine similarity
    # Returns a list of document ids in rank order
    def rank_documents(self, cosine_simularity: dict):
        return sorted(cosine_simularity, key= cosine_simularity.get,  reverse=True)
    
    #==============================================================================
    # Vector Space Model

    # Method for computing the query and document vectors
    # Returns a dictionary of the query vector and a dictionary of the document vectors
    # document_vectors = {doc_id: {term: weight}}
    # query_vector = {term: weight}
    def compute_vectors(self, query: list):
        # Initialise the query vector and the document vectors
        document_vectors = {}
        query_vector = {}

        # For each term in the query
        query_set = set(query)
        for term in query_set:
            # If the term is not in the index, continue (Skipping unkown terms)
            if term not in self.index:
                continue

            query_vector[term] = self.get_query_weighting(term, query)

            relevant_docs = self.index[term]
            for doc in relevant_docs:
                if doc not in document_vectors:
                    document_vectors[doc] = {}
                document_vectors[doc][term] = self.get_weighting(term, doc)

        # Return the query vector and the document vectors
        return query_vector, document_vectors
    
    #==============================================================================
    # Compare Query and Document
        
    # Method for finding the cosine similarity between a query and a document
    # Returns the cosine similarity between the two vectors
    def cosine_similarity(self, query_vector: dict , doc_vector: dict, doc_id: int):
        # Compute the numerator
        numerator = 0
        numerator = sum([query_vector[term] * doc_vector[term] 
             for term in query_vector if term in doc_vector])
        # Get the denominator from the document vector
        denominator = self.document_vectors_length[doc_id]

        # Return the cosine similarity
        return numerator / denominator
    
    #==============================================================================
    # Weighting Schemes

    # Method for computing the query tf term weighting
    def compute_query_tf(self, term: str, query: list):
        return query.count(term)
    
    # Method for computing the query tfidf term weighting
    def compute_query_tfidf(self, term: str, query: list):
       return self.compute_query_tf(term, query) * self.compute_idf(term)
    
    # Method for deciding the query term weighting
    def get_query_weighting(self, term: str, query: list):
        if self.term_weighting == "binary":
            return 1
        elif self.term_weighting == 'tf':
            return self.compute_query_tf(term, query)
        elif self.term_weighting == 'tfidf':
            return self.compute_query_tfidf(term, query)

    # Method for computing the document tf term weighting      
    def compute_tf(self, term: str, doc_id: int):
        return self.index[term][doc_id]

    # Method for computing the document idf term weighting 
    def compute_idf(self, term: str):
        return math.log(self.num_docs / len(self.index[term]))
    
    # Method for computing the document tfidf term weighting
    def compute_tf_idf(self, term: str, doc_id: int):
        return self.index[term][doc_id] * self.compute_idf(term)

    # Method for deciding the document term weighting
    def get_weighting(self, term: str, doc_id: int):
        if self.term_weighting == "binary":
            return 1
        elif self.term_weighting  == "tf":
            return self.compute_tf(term, doc_id)
        elif self.term_weighting  == "tfidf":
            return self.compute_tf_idf(term, doc_id)