
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
        self.doc_term_matrix = self.compute_doc_term_matrix()
        
    
    def compute_number_of_documents(self): 
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    #==============================================================================
    # Query Processing

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). ​Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        matrix = self.doc_term_matrix
        query_vector = self.compute_query_vector(query)
        scores = {}
        
        for doc_id in matrix:
            scores[doc_id] = self.cosine_similarity(query_vector, matrix[doc_id])

        ranked_docs = self.rank_documents(scores)

        return ranked_docs
    
    #==============================================================================
    # Rank Results

    # Method ranking the documents based on the cosine similarity
    # Returns a list of document ids in rank order
    def rank_documents(self, cosine_simularity: dict):
        return sorted(cosine_simularity, key= cosine_simularity.get,  reverse=True)[:10]

    # Method performing binary comperation for a single query
    # Returns a m x n document-term matrix where:a
    # m is the number of documents in the collection
    # n is the number of terms in the query
    # The values in the matrix are the frequency of the term in the document
    # For each vector, we also compute the denominator for the cosine similarity
    def compute_doc_term_matrix(self):
        # Initialise the document-term matrix
        doc_term_matrix = {}
       
        # For each document in the collection
        for doc_id in self.doc_ids:

            # Initialise the document vector    
            doc = {}
            # Initialise the denominator for the cosine similarity
            doc['ENUMERATOR'] = 0
           
            # For each term in the query
            for term in self.index:
                # If the term is in the document, add the frequency
                if doc_id in self.index[term]:
                    doc[term] = self.index[term][doc_id] * self.get_weighting(term, doc_id)
                    # Compute the denominator for the cosine similarity
                    doc['ENUMERATOR'] += doc[term] **  2
                
                # If the term is not in the document, add 0
                else:
                    doc[term] = 0
            
            # Compute the denominator for the cosine similarity
            doc['DENOMINATOR'] = math.sqrt(doc['ENUMERATOR'])
            # Add the document vector to the document-term matrix
            doc_term_matrix[doc_id] = doc
        return doc_term_matrix
    


    # Method for finding the term frequency for a single query
    # Returns t vectore of term frequencies:
    def compute_query_vector(self, query: list):
        # Initialise the query set and the query vector
        query_set = set(query)
        query_vector = {}

        # For each term in the query
        for term in query_set:
            # If the term is not in the index, continue
            # (Skipping unkown terms)
            if term not in self.index:
                continue

            # If the term is not in the query vector, add it
            if query_vector.get(term) == None:
                query_vector[term] = self.get_query_weighting(term, query)

            # Else, increment the frequency
            else:
                query_vector[term] += self.get_query_weighting(term, query)

        # Return the query vector
        return query_vector
    
    #==============================================================================
    # Compare Query and Document

        
    # Method for finding the cosine similarity between a query and a document
    # Returns the cosine similarity between the two vectors
    def cosine_similarity(self, query_vec: dict , doc_vec: dict):

        # Initialise the numerator
        numerator = 0
        query_denominator = 0

        # Compute the numerator
        for term in query_vec:
            numerator += query_vec[term] * doc_vec[term]
            query_denominator += query_vec[term] ** 2
        
        # Get the denominator from the document vector
        denominator = doc_vec['DENOMINATOR']

        # Return the cosine similarity
        return numerator / denominator
    
    #==============================================================================
    # Weighting Schemes

    def compute_query_tf(self, term: str, query: list):
        return query.count(term)
    
    def compute_query_tfidf(self, term: str, query: list):
       return self.compute_query_tf(term, query) * self.compute_idf(term)
    
    def get_query_weighting(self, term: str, query: list):
        if self.term_weighting == "binary":
            return 1
        elif self.term_weighting == 'tf':
            return self.compute_query_tf(term, query)
        elif self.term_weighting == 'tfidf':
            return self.compute_query_tfidf(term, query)
                
    def compute_tf(self, term: str, doc_id: int):
        return self.index[term][doc_id]
      
    def compute_idf(self, term: str):
        return math.log(self.num_docs / len(self.index[term]))
    
    def compute_tf_idf(self, term: str, doc_id: int):
        return self.index[term][doc_id] * self.compute_idf(term)

    def get_weighting(self, term: str, doc_id: int):
        if self.term_weighting == "binary":
            return 1
        elif self.term_weighting  == "tf":
            return self.compute_tf(term, doc_id)
        elif self.term_weighting  == "tfidf":
            return self.compute_tf_idf(term, doc_id)