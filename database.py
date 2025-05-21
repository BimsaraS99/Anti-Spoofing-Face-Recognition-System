import sqlite3
import numpy as np
from typing import Optional, Dict, List, Tuple


class EmbeddingDatabase:
    """SQLite database handler for storing and retrieving named embeddings."""
    
    def __init__(self, db_path: str = "embeddings.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    

    def _initialize_db(self) -> None:
        """Create database table if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON embeddings(name)")
            self.conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Database initialization failed: {str(e)}")
    

    def save_embedding(self, name: str, embedding: np.ndarray) -> bool:
        """
        Save an embedding with a unique name.
        
        Args:
            name: Unique identifier for the embedding
            embedding: Numpy array containing the embedding
            
        Returns:
            True if successful, False if name already exists
        """
        try:
            embedding_bytes = embedding.astype(np.float32).tobytes()
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO embeddings (name, embedding) VALUES (?, ?)",
                (name, embedding_bytes)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Name already exists
        except Exception as e:
            raise RuntimeError(f"Failed to save embedding: {str(e)}")
    

    def get_embedding(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding by name.
        
        Args:
            name: Name of the embedding to retrieve
            
        Returns:
            Numpy array if found, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE name = ?",
                (name,)
            )
            result = cursor.fetchone()
            if result:
                return np.frombuffer(result[0], dtype=np.float32)
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embedding: {str(e)}")
    

    def update_embedding(self, name: str, new_embedding: np.ndarray) -> bool:
        """
        Update an existing embedding.
        
        Args:
            name: Name of the embedding to update
            new_embedding: New embedding values
            
        Returns:
            True if updated, False if name doesn't exist
        """
        try:
            new_bytes = new_embedding.astype(np.float32).tobytes()
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE embeddings SET embedding = ? WHERE name = ?",
                (new_bytes, name)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            raise RuntimeError(f"Failed to update embedding: {str(e)}")
    

    def delete_embedding(self, name: str) -> bool:
        """
        Delete an embedding by name.
        
        Args:
            name: Name of the embedding to delete
            
        Returns:
            True if deleted, False if name doesn't exist
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM embeddings WHERE name = ?",
                (name,)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            raise RuntimeError(f"Failed to delete embedding: {str(e)}")
    

    def list_all(self) -> Dict[str, np.ndarray]:
        """Return all embeddings as {name: embedding} dictionary."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, embedding FROM embeddings")
            return {
                name: np.frombuffer(emb, dtype=np.float32)
                for name, emb in cursor.fetchall()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to list embeddings: {str(e)}")
    

    def find_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Embedding to compare against
            top_k: Number of most similar results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (name, similarity_score) tuples
        """
        try:
            from scipy.spatial.distance import cosine
            
            embeddings = self.list_all()
            similarities = [
                (name, 1 - cosine(query_embedding, emb))
                for name, emb in embeddings.items()
            ]
            filtered = [x for x in similarities if x[1] >= threshold]
            return sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
        except ImportError:
            raise ImportError("scipy is required for similarity search")
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {str(e)}")
    

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    

    def __enter__(self):
        """Context manager entry."""
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
