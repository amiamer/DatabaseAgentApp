import streamlit as st
import pandas as pd
import pyodbc
import json
import time
import os
import gc
import numpy as np
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from typing import List, Dict, Optional
import hashlib

load_dotenv()

st.set_page_config(page_title="Database Agent", page_icon="🤖", layout="wide")

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
VECTOR_DB_PATH = Path("./vector_db")

class SimpleVectorStore:
    def __init__(self, embedding_function, persist_directory=None):
        self.embedding_function = embedding_function
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.documents = {}
        self.embeddings = {}
        self.doc_ids = []
        
        if self.persist_directory and self.persist_directory.exists():
            self.load()
    
    def add_documents(self, documents, ids):
        for doc, doc_id in zip(documents, ids):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            embedding = self.embedding_function.embed_query(content)
            
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata,
                'timestamp': time.time()
            }
            self.embeddings[doc_id] = embedding
            
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
        
        if self.persist_directory:
            self.save()
    
    def similarity_search(self, query, k=5):
        if not self.documents:
            return []
        
        query_embedding = self.embedding_function.embed_query(query)
        similarities = []
        
        for doc_id in self.doc_ids:
            if doc_id in self.embeddings:
                doc_embedding = self.embeddings[doc_id]
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((doc_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, similarity in similarities[:k]:
            doc_data = self.documents[doc_id]
            result = type('Document', (), {
                'page_content': doc_data['content'],
                'metadata': {**doc_data['metadata'], 'similarity': similarity}
            })()
            results.append(result)
        
        return results
    
    def _cosine_similarity(self, vec1, vec2):
        a = np.array(vec1)
        b = np.array(vec2)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    
    def save(self):
        if not self.persist_directory:
            return
        self.persist_directory.mkdir(exist_ok=True)
        
        with open(self.persist_directory / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        with open(self.persist_directory / "embeddings.pkl", 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        with open(self.persist_directory / "doc_ids.json", 'w') as f:
            json.dump(self.doc_ids, f)
    
    def load(self):
        try:
            if (self.persist_directory / "documents.json").exists():
                with open(self.persist_directory / "documents.json", 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            if (self.persist_directory / "embeddings.pkl").exists():
                with open(self.persist_directory / "embeddings.pkl", 'rb') as f:
                    self.embeddings = pickle.load(f)
            
            if (self.persist_directory / "doc_ids.json").exists():
                with open(self.persist_directory / "doc_ids.json", 'r') as f:
                    self.doc_ids = json.load(f)
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            self.documents = {}
            self.embeddings = {}
            self.doc_ids = []
    
    def get_stats(self):
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }

class DatabaseAgent:
    def __init__(self):
        self.connection = None
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.schema_data = None
        self.connection_params = None
    
    def initialize_ai_models(self):
        try:
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            test_embed = self.embeddings.embed_query("test")
            if not test_embed:
                return False, "Embeddings test failed"
            
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return False, "ANTHROPIC_API_KEY not found"
            
            self.llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
            return True, "AI models initialized successfully"
            
        except Exception as e:
            return False, f"AI initialization failed: {e}"
    
    def connect_database(self, server, database, username, password):
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};DATABASE={database};"
                f"UID={username};PWD={password};"
                f"Connection Timeout=30;Command Timeout=60;"
                "Encrypt=yes;TrustServerCertificate=yes;"
            )
            self.connection = pyodbc.connect(conn_str)
            self.connection_params = {
                'server': server, 'database': database,
                'username': username, 'password': password
            }
            return True, "Database connected successfully"
        except Exception as e:
            return False, f"Database connection failed: {e}"
    
    def extract_schema(self, database_name):
        if not self.connection:
            return None, "No database connection"
        
        try:
            # Extract tables and views
            tables_query = """
                SELECT TABLE_SCHEMA as [Schema], TABLE_NAME as [Table], TABLE_TYPE as [Type]
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW') 
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
            tables_df = pd.read_sql(tables_query, self.connection)
            
            # Extract columns
            columns_query = """
                SELECT 
                    TABLE_SCHEMA as [Schema],
                    TABLE_NAME as [Table],
                    COLUMN_NAME as [Column],
                    DATA_TYPE as [DataType],
                    IS_NULLABLE as [Nullable],
                    COLUMN_DEFAULT as [DefaultValue],
                    CHARACTER_MAXIMUM_LENGTH as [MaxLength],
                    NUMERIC_PRECISION as [Precision],
                    ORDINAL_POSITION as [Position]
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """
            columns_df = pd.read_sql(columns_query, self.connection)
            
            # Extract foreign keys
            fk_query = """
                SELECT fk.name AS [FK_Name], tp.name AS [Parent_Table], cp.name AS [Parent_Column],
                       tr.name AS [Referenced_Table], cr.name AS [Referenced_Column]
                FROM sys.foreign_keys fk
                INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
                INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
                INNER JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
                INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
                ORDER BY tp.name, tr.name
            """
            fk_df = pd.read_sql(fk_query, self.connection)
            
            # Extract stored procedures
            sp_query = """
                SELECT TOP 50
                    ROUTINE_SCHEMA as [Schema], 
                    ROUTINE_NAME as [Procedure_Name],
                    LEFT(ROUTINE_DEFINITION, 2000) as [Definition]
                FROM INFORMATION_SCHEMA.ROUTINES 
                WHERE ROUTINE_TYPE = 'PROCEDURE' 
                    AND ROUTINE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
                ORDER BY ROUTINE_SCHEMA, ROUTINE_NAME
            """
            sp_df = pd.read_sql(sp_query, self.connection)
            
            # Build schema data
            self.schema_data = {
                "database": database_name,
                "extraction_time": pd.Timestamp.now().isoformat(),
                "tables_views": tables_df.to_dict(orient="records"),
                "columns": columns_df.to_dict(orient="records"),
                "foreign_keys": fk_df.to_dict(orient="records"),
                "stored_procedures": sp_df.to_dict(orient="records"),
                "summary": {
                    "total_tables": len(tables_df[tables_df['Type'] == 'BASE TABLE']),
                    "total_views": len(tables_df[tables_df['Type'] == 'VIEW']),
                    "total_columns": len(columns_df),
                    "total_procedures": len(sp_df),
                    "total_foreign_keys": len(fk_df)
                }
            }
            
            return self.schema_data, "Schema extracted successfully"
            
        except Exception as e:
            return None, f"Schema extraction failed: {e}"
    
    def create_chunks(self, schema_data):
        chunks = []
        database_name = schema_data.get('database', 'Unknown')
        
        # Database overview chunk
        summary = schema_data.get('summary', {})
        overview_content = f"""DATABASE: {database_name}
SUMMARY:
- Tables: {summary.get('total_tables', 0)}
- Views: {summary.get('total_views', 0)}
- Columns: {summary.get('total_columns', 0)}
- Stored Procedures: {summary.get('total_procedures', 0)}
- Foreign Keys: {summary.get('total_foreign_keys', 0)}

This database contains structured business data across multiple tables with established relationships."""
        
        chunks.append({
            'id': f"{database_name}_overview",
            'content': overview_content,
            'metadata': {
                'type': 'overview',
                'database': database_name
            }
        })
        
        # Table chunks
        tables = [t for t in schema_data.get('tables_views', []) if t.get('Type') == 'BASE TABLE']
        for table in tables[:50]:  # Limit tables
            schema_name = table.get('Schema', 'dbo')
            table_name = table.get('Table', 'Unknown')
            
            # Get columns for this table
            table_columns = [
                col for col in schema_data.get('columns', [])
                if col.get('Table') == table_name and col.get('Schema') == schema_name
            ]
            
            # Create table description
            content = f"""TABLE: {schema_name}.{table_name}
COLUMNS ({len(table_columns)} total):
"""
            for col in table_columns[:20]:  # Limit columns shown
                nullable = "NULL" if col.get('Nullable') == 'YES' else "NOT NULL"
                content += f"- {col.get('Column')} ({col.get('DataType')}) {nullable}\n"
            
            # Infer table purpose
            table_lower = table_name.lower()
            if any(word in table_lower for word in ['customer', 'client', 'user']):
                purpose = "Customer/User data storage"
            elif any(word in table_lower for word in ['order', 'sale', 'transaction']):
                purpose = "Transaction/Order data"
            elif any(word in table_lower for word in ['product', 'item', 'inventory']):
                purpose = "Product/Inventory data"
            elif any(word in table_lower for word in ['employee', 'staff']):
                purpose = "Employee/HR data"
            else:
                purpose = "Business data storage"
            
            content += f"\nPURPOSE: {purpose}"
            
            chunks.append({
                'id': f"{database_name}_table_{schema_name}_{table_name}",
                'content': content,
                'metadata': {
                    'type': 'table',
                    'schema': schema_name,
                    'table': table_name,
                    'database': database_name,
                    'column_count': len(table_columns)
                }
            })
        
        # Relationships chunk
        relationships = schema_data.get('foreign_keys', [])
        if relationships:
            content = f"FOREIGN KEY RELATIONSHIPS:\n"
            content += f"Total relationships: {len(relationships)}\n\n"
            
            for fk in relationships[:30]:  # Limit relationships
                parent = fk.get('Parent_Table', 'Unknown')
                referenced = fk.get('Referenced_Table', 'Unknown')
                parent_col = fk.get('Parent_Column', 'Unknown')
                ref_col = fk.get('Referenced_Column', 'Unknown')
                content += f"{parent}.{parent_col} -> {referenced}.{ref_col}\n"
            
            content += "\nThese relationships define how tables are connected and can be used for JOIN operations in SQL queries."
            
            chunks.append({
                'id': f"{database_name}_relationships",
                'content': content,
                'metadata': {
                    'type': 'relationships',
                    'database': database_name,
                    'relationship_count': len(relationships)
                }
            })
        
        # Stored procedures chunk
        procedures = schema_data.get('stored_procedures', [])
        if procedures:
            content = f"STORED PROCEDURES:\n"
            content += f"Total procedures: {len(procedures)}\n\n"
            
            for proc in procedures:
                schema_name = proc.get('Schema', 'dbo')
                proc_name = proc.get('Procedure_Name', 'Unknown')
                definition = proc.get('Definition', '')[:500]  # Limit definition length
                
                content += f"PROCEDURE: {schema_name}.{proc_name}\n"
                if definition:
                    content += f"Definition snippet: {definition}...\n\n"
            
            chunks.append({
                'id': f"{database_name}_procedures",
                'content': content,
                'metadata': {
                    'type': 'procedures',
                    'database': database_name,
                    'procedure_count': len(procedures)
                }
            })
        
        return chunks
    
    def create_vector_store(self, chunks):
        if not self.embeddings:
            return False, "Embeddings not initialized"
        
        try:
            self.vector_store = SimpleVectorStore(
                embedding_function=self.embeddings,
                persist_directory=str(VECTOR_DB_PATH)
            )
            
            # Convert chunks to documents
            documents = []
            ids = []
            
            for chunk in chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata=chunk['metadata']
                )
                documents.append(doc)
                ids.append(chunk['id'])
            
            # Add to vector store
            self.vector_store.add_documents(documents, ids)
            
            stats = self.vector_store.get_stats()
            return True, f"Vector store created with {stats['total_documents']} documents"
            
        except Exception as e:
            return False, f"Vector store creation failed: {e}"
    
    def query_with_rag(self, user_question):
        if not self.vector_store or not self.llm:
            return None, None, "Vector store or LLM not initialized"
        
        try:
            # Retrieve relevant chunks
            relevant_docs = self.vector_store.similarity_search(user_question, k=5)
            
            if not relevant_docs:
                return None, None, "No relevant information found"
            
            # Build context
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"CONTEXT {i+1}:\n{doc.page_content}\n\n"
            
            # Create RAG prompt
            prompt = f"""You are a SQL expert. Based on the database schema information provided, generate a SQL query to answer the user's question.

DATABASE CONTEXT:
{context}

USER QUESTION: {user_question}

Instructions:
1. Generate a SQL query that answers the question
2. Use proper SQL Server syntax with square brackets for table/column names
3. Include schema names: [schema].[table]
4. Add appropriate WHERE clauses, JOINs, and ORDER BY as needed
5. Limit results to TOP 100 unless specifically asked for more
6. If the question asks for a report, create a well-structured query with proper aggregations
7. If the question asks for a stored procedure, provide the CREATE PROCEDURE syntax
8. If the question asks for a function, provide the CREATE FUNCTION syntax

Return ONLY the SQL query, nothing else."""

            # Generate SQL
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Clean up response
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            return sql_query, relevant_docs, None
            
        except Exception as e:
            return None, None, f"RAG query failed: {e}"
    
    def execute_query(self, query):
        if not self.connection:
            return None, "No database connection"
        
        try:
            start_time = time.time()
            
            # Check if it's a DDL statement (CREATE PROCEDURE, CREATE FUNCTION, etc.)
            query_upper = query.upper().strip()
            is_ddl = any(query_upper.startswith(cmd) for cmd in ['CREATE', 'ALTER', 'DROP'])
            
            if is_ddl:
                # Execute DDL statement
                cursor = self.connection.cursor()
                cursor.execute(query)
                self.connection.commit()
                cursor.close()
                
                execution_time = time.time() - start_time
                return {
                    'success': True,
                    'message': 'DDL statement executed successfully',
                    'execution_time': execution_time,
                    'rows_affected': 0
                }, None
            else:
                # Execute SELECT query
                df = pd.read_sql(query, self.connection)
                execution_time = time.time() - start_time
                
                return {
                    'success': True,
                    'data': df,
                    'execution_time': execution_time,
                    'rows_affected': len(df)
                }, None
                
        except Exception as e:
            return None, f"Query execution failed: {e}"

def main():
    st.title("Database Agent - Natural Language to SQL")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = DatabaseAgent()
    
    agent = st.session_state.agent
    
    # Auto-initialize AI models on first run
    if not st.session_state.get('ai_initialized'):
        with st.spinner("Initializing AI models..."):
            success, message = agent.initialize_ai_models()
            if success:
                st.session_state.ai_initialized = True
            else:
                st.error(f"AI initialization failed: {message}")
                st.stop()
    
    # Sidebar for database connection
    with st.sidebar:
        st.header("Database Connection")
        with st.form("db_connection"):
            server = st.text_input("Server", value="localhost")
            database = st.text_input("Database")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Connect"):
                # Connect to database
                with st.spinner("Connecting to database..."):
                    success, message = agent.connect_database(server, database, username, password)
                
                if success:
                    st.success(message)
                    st.session_state.db_connected = True
                    
                    # Auto-extract and process schema
                    with st.spinner("Extracting database schema..."):
                        schema_data, schema_message = agent.extract_schema(database)
                    
                    if schema_data:
                        st.success(schema_message)
                        
                        with st.spinner("Creating chunks..."):
                            chunks = agent.create_chunks(schema_data)
                            st.success(f"Created {len(chunks)} chunks")
                        
                        with st.spinner("Building vector store..."):
                            vs_success, vs_message = agent.create_vector_store(chunks)
                            if vs_success:
                                st.success(vs_message)
                                st.session_state.vector_store_ready = True
                                st.rerun()
                            else:
                                st.error(vs_message)
                    else:
                        st.error(schema_message)
                else:
                    st.error(message)
    
    # Main query interface
    if st.session_state.get('vector_store_ready'):
        st.success("System ready! Ask questions about your database.")
        
        # Sample queries
        with st.expander("Sample Queries"):
            st.write("**Data Retrieval:**")
            st.write("- Show me all customers who placed orders last month")
            st.write("- List top 10 products by sales")
            st.write("- Find customers with no recent orders")
            
            st.write("**Reports:**")
            st.write("- Generate a sales report by region")
            st.write("- Show monthly revenue trends")
            st.write("- Create customer activity summary")
            
            st.write("**Database Objects:**")
            st.write("- Create a stored procedure to get customer orders")
            st.write("- Make a function to calculate order total")
            st.write("- Generate a view for active customers")
        
        # Query input
        user_question = st.text_area(
            "Ask your question:",
            placeholder="Show me sales by region for the last quarter",
            height=100
        )
        
        if st.button("Execute Query", type="primary") and user_question:
            with st.spinner("Processing question with RAG..."):
                sql_query, context_docs, error = agent.query_with_rag(user_question)
            
            if error:
                st.error(error)
            elif sql_query:
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")
                
                # Show context used
                with st.expander("Context Used"):
                    for i, doc in enumerate(context_docs):
                        st.write(f"**Context {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                
                # Execute query
                with st.spinner("Executing query..."):
                    result, error = agent.execute_query(sql_query)
                
                if error:
                    st.error(error)
                elif result:
                    if result['success']:
                        st.success(f"Query executed in {result['execution_time']:.2f} seconds")
                        
                        if 'data' in result and result['data'] is not None:
                            df = result['data']
                            st.subheader("Results")
                            st.write(f"Returned {len(df)} rows")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download Results as CSV",
                                csv,
                                f"query_results_{int(time.time())}.csv",
                                "text/csv"
                            )
                        else:
                            st.info(result.get('message', 'Query executed successfully'))
    else:
        st.info("Please complete the setup steps in the sidebar to begin.")

if __name__ == "__main__":
    main()