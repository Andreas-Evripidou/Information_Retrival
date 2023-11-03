
import math


class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. ​(You can extend this method, as required.)
    def __init__(self,index,term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.cf_dict = self.compute_cf()
        self.df_dict = self.compute_df()
        self.doc_term_matrix = self.compute_doc_term_matrix()
        # self.idf_dict = self.compute_idf()
        # write self.index to file
        open("help/index.txt", "w").write(str(self.index))
        # write self.num_docs to file
        open("help/num_docs.txt", "w").write(str(self.num_docs))
        
        
    def compute_number_of_documents(self): 
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). ​Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        query_set = set(query)
        matrix = self.doc_term_matrix
        scores = {}
        query_vector = self.compute_query_vector(query, query_set)
        
        for i, doc_vector in enumerate(matrix):
            # print("For document:", i, "and vector:", doc_vector)
            scores[i] = self.cosine_similarity(query_vector, doc_vector)
        


        ranked_docs = self.rank_documents(scores)
        print("Ranked docs:", ranked_docs)
        # add 1 to each doc id to match the qrels file
        ranked_docs = [doc_id + 1 for doc_id in ranked_docs]

        return ranked_docs

    # Method performing binary comperation for a single query
    # Returns a m x n document-term matrix where:
    # m is the number of documents in the collection
    # n is the number of terms in the query
    # The values in the matrix are the frequency of the term in the document
    # For each vector, we also compute the denominator for the cosine similarity
    def compute_doc_term_matrix(self):
        # Initialise the document-term matrix
        doc_term_matrix = []
       
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
                    doc[term] = self.index[term][doc_id]
                    # Compute the denominator for the cosine similarity
                    doc['ENUMERATOR'] += self.index[term][doc_id] ** 2
                
                # If the term is not in the document, add 0
                else:
                    doc[term] = 0
            
            # Compute the denominator for the cosine similarity
            doc['DENOMINATOR'] = math.sqrt(doc['ENUMERATOR'])
            # Add the document vector to the document-term matrix
            doc_term_matrix.append(doc)
        return doc_term_matrix
    
    # Method for finding the cosine similarity between a query and a document
    # Returns the cosine similarity between the two vectors
    def cosine_similarity(self, query_vec, doc_vec):

        numerator = 0
        doc_denominator = doc_vec['DENOMINATOR']
        for term in query_vec:
            numerator += query_vec[term] * doc_vec[term]


        if doc_denominator == 0:
            return -100
        
        return numerator / doc_denominator

    # Method for finding the term frequency for a single query
    # Returns t vectore of term frequencies:
    def compute_query_vector(self, query, query_set):
        query_vector = {}
        for term in query_set:
            if term not in self.index:
                continue
            if query_vector.get(term) == None:
                query_vector[term] = 1
            else:
                query_vector[term] += 1
        return query_vector
    
    # Method ranking the documents based on the cosine similarity
    # Returns a list of document ids in rank order
    def rank_documents(self, cosine_simularity):
        return sorted(cosine_simularity, key= cosine_simularity.get,  reverse=True)[:10]

    
    # Method to compute the icollection frequency
    # Returns a list of collection frequencies for each term
    def compute_cf(self):
        cf = {}
        for term in self.index:
            cf[term] = sum(self.index[term].values())
        return cf
    
    # Method to compute the document frequency
    # Returns a list of document frequencies for each term
    def compute_df(self):
        df = {}
        for term in self.index:
            df[term] = len(self.index[term])
        return df

    