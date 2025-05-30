# Database Agent

Converts natural language to SQL queries for MSSQL databases.

## Requirements

- Python 3.8+
- MSSQL Server access
- ODBC Driver 18 for SQL Server
- Ollama with nomic-embed-text model
- Anthropic API key

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install and start Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama serve
   ```

3. Create `.env` file:
   ```
   ANTHROPIC_API_KEY=your_api_key
   ```

4. Run application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter database credentials in sidebar
2. Click Connect
3. Ask questions in natural language
4. Review generated SQL
5. Execute and view results

## Example Queries

- "Show all customers from last month"
- "Top 10 products by sales"
- "Create sales report by region"
- "Generate stored procedure for customer orders"

## Troubleshooting

**Connection fails**: Check ODBC driver, credentials, server access
**AI initialization fails**: Verify Ollama is running and API key is set
**Slow performance**: First query loads models, subsequent queries are faster

## Files

- `app.py` - Main application
- `requirements.txt` - Dependencies
- `.env` - API keys (create this)
- `vector_db/` - Auto-generated vector store
- `sample_queries.yaml` - Auto-generated query samples
