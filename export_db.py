import os
import pandas as pd
import json
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters from environment variables
db_params = {
    'host': os.environ.get('PGHOST'),
    'port': os.environ.get('PGPORT'),
    'database': os.environ.get('PGDATABASE'),
    'user': os.environ.get('PGUSER'),
    'password': os.environ.get('PGPASSWORD')
}

# Connect to the database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Get list of all tables
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name
""")
tables = [table[0] for table in cursor.fetchall()]

# Export data and schema for each table
export_data = {}

for table in tables:
    print(f"Exporting table: {table}")
    
    # Get table schema
    cursor.execute(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
    """)
    schema = [{"column": col[0], "type": col[1], "nullable": col[2]} for col in cursor.fetchall()]
    
    # Get table data
    cursor.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cursor.description]
    
    rows = []
    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))
    
    # Store schema and data
    export_data[table] = {
        "schema": schema,
        "data": rows
    }

# Export to JSON file
with open('database_export.json', 'w') as f:
    json.dump(export_data, f, indent=2, default=str)

print("Export completed. File saved as 'database_export.json'")

# Close the connection
cursor.close()
conn.close()