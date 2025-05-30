# Database Agent - Natural Language to SQL

A powerful web application that converts natural language questions into SQL queries and executes them against your MSSQL database. Built with AI-powered understanding of your database schema.

## Features

- 🤖 **Natural Language Processing**: Ask questions in plain English
- 🗄️ **Automatic Schema Discovery**: Extracts tables, columns, relationships, and stored procedures
- ⚡ **Smart Query Generation**: Uses AI to generate optimized SQL queries
- 🔍 **RAG Architecture**: Retrieves relevant schema context for better query generation
- 📊 **Results Display**: Shows query results in user-friendly tables
- 💾 **Export Data**: Download results as CSV files
- 🔧 **Advanced SQL**: Generates reports, stored procedures, and functions

## Prerequisites

Before installing, ensure you have:

1. **Python 3.8+** installed on your system
2. **MSSQL Server** with a database you want to query
3. **ODBC Driver 18 for SQL Server** installed
4. **Ollama** installed and running locally with the `nomic-embed-text` model
5. **Claude API access** (Anthropic API key)

## Installation

1. **Clone or download** the project files to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.ai/download
   
   # Pull the required embedding model
   ollama pull nomic-embed-text
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project directory:
   ```
   ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

5. **Install ODBC Driver** (if not already installed):
   - **Windows**: Download from Microsoft's official website
   - **macOS**: `brew install msodbcsql18`
   - **Linux**: Follow Microsoft's Linux installation guide

## Running the Application

1. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

2. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### 1. Connect to Database
- Enter your MSSQL server details in the sidebar
- Click "Connect" to establish connection
- The app will automatically extract and process your database schema

### 2. Ask Questions
Once connected, you can ask questions like:

**Data Retrieval:**
- "Show me all customers who placed orders last month"
- "List top 10 products by sales"
- "Find customers with no recent orders"

**Reports:**
- "Generate a sales report by region"
- "Show monthly revenue trends"
- "Create customer activity summary"

**Database Objects:**
- "Create a stored procedure to get customer orders"
- "Make a function to calculate order total"
- "Generate a view for active customers"

### 3. Review and Execute
- The app generates SQL queries based on your questions
- Review the generated SQL before execution
- View results in formatted tables
- Download results as CSV if needed

## Project Structure

```
database-agent/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # This file
├── vector_db/            # Vector store persistence (auto-created)
└── sample_queries.yaml   # Sample queries storage (auto-created)
```

## Configuration

### Database Connection
The app supports MSSQL Server with these connection parameters:
- Server address
- Database name
- Username and password
- Encrypted connections (TrustServerCertificate=yes)

### AI Models
- **Embeddings**: Uses Ollama's `nomic-embed-text` model for semantic search
- **LLM**: Uses Claude 3.5 Sonnet for SQL generation

## Troubleshooting

### Common Issues

1. **"AI initialization failed"**
   - Ensure Ollama is running: `ollama serve`
   - Check if the embedding model is installed: `ollama list`
   - Verify your Anthropic API key in the `.env` file

2. **"Database connection failed"**
   - Verify ODBC Driver 18 is installed
   - Check server address, database name, and credentials
   - Ensure the database server allows remote connections

3. **"No module named 'pyodbc'"**
   - Install missing dependencies: `pip install -r requirements.txt`

4. **Slow query generation**
   - This is normal for the first query as the system loads models
   - Subsequent queries should be faster

### Performance Tips

- The vector store persists between sessions for faster startup
- Limit result sets with appropriate WHERE clauses
- Complex schemas may take longer to process initially

## Security Notes

- Store database credentials securely
- Use read-only database accounts when possible
- The app uses encrypted connections to MSSQL
- API keys are stored in environment variables

## Contributing

To improve the application:
1. Add sample queries through the sidebar interface
2. Report issues or suggest features
3. Extend functionality for other database types

## License

This project is provided as-is for educational and commercial use.

---

**Need Help?** Check the troubleshooting section or ensure all prerequisites are properly installed.
