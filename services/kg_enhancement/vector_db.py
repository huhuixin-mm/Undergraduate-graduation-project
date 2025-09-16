"""
Vector Database Handler for document retrieval
Handles embedding storage and similarity search
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings


class VectorDBHandler:
    """Handler for vector database operations"""
    
    def __init__(self, 
                 persist_directory: str = None,
                 embedding_model: str = None,
                 collection_name: str = None):
        
        self.persist_directory = persist_directory or settings.vector_db.chroma_persist_directory
        self.embedding_model_name = embedding_model or settings.vector_db.embedding_model
        self.collection_name = collection_name or settings.vector_db.collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Medical documents and knowledge"}
            )
            
            print(f"✅ Vector database initialized: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]] = None,
                     ids: List[str] = None) -> bool:
        """
        Add documents to the vector database
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        
        Returns:
            bool: True if successful
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         score_threshold: float = 0.7) -> List[Tuple[Any, float]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of (document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Process results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= score_threshold:
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        
                        doc_obj = type('Document', (), {
                            'page_content': doc,
                            'metadata': metadata
                        })()
                        
                        documents.append((doc_obj, similarity))
            
            return documents
            
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=ids)
            print(f"✅ Deleted {len(ids)} documents")
            return True
        except Exception as e:
            print(f"❌ Failed to delete documents: {e}")
            return False
    
    def update_documents(self, 
                        ids: List[str],
                        documents: List[str] = None,
                        metadatas: List[Dict[str, Any]] = None) -> bool:
        """Update existing documents"""
        try:
            update_kwargs = {"ids": ids}
            
            if documents:
                embeddings = self.embedding_model.encode(documents).tolist()
                update_kwargs.update({
                    "documents": documents,
                    "embeddings": embeddings
                })
            
            if metadatas:
                update_kwargs["metadatas"] = metadatas
            
            self.collection.update(**update_kwargs)
            print(f"✅ Updated {len(ids)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update documents: {e}")
            return False


# Example document processing utilities
class DocumentProcessor:
    """Utility class for processing documents before adding to vector DB"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF file
        Note: Requires PyPDF2 or similar library
        """
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            return text
            
        except ImportError:
            print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"❌ Failed to extract from PDF: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)]', ' ', text)
        
        # Normalize case (optional)
        # text = text.lower()
        
        return text.strip()


# Factory function
def create_vector_db() -> VectorDBHandler:
    """Create and return a VectorDBHandler instance"""
    return VectorDBHandler()


if __name__ == "__main__":
    # Test the vector database
    vector_db = create_vector_db()
    
    # Test document addition
    test_documents = [
        "The brain extracellular space (ECS) is a narrow, tortuous space between brain cells.",
        "Traditional Chinese Medicine views the brain as connected to kidney essence.",
        "Neuronal communication occurs through synaptic transmission in the ECS."
    ]
    
    test_metadata = [
        {"source": "neuroscience_paper", "type": "definition"},
        {"source": "tcm_text", "type": "traditional_knowledge"},
        {"source": "physiology_book", "type": "mechanism"}
    ]
    
    success = vector_db.add_documents(test_documents, test_metadata)
    
    if success:
        print("\n🔍 Testing similarity search:")
        query = "How does brain extracellular space work?"
        results = vector_db.similarity_search(query, k=3)
        
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1} (score: {score:.3f}): {doc.page_content}")
        
        print(f"\n📊 Collection stats: {vector_db.get_collection_stats()}")