import os
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

# Create SQL file for schema and data
with open('database_schema.sql', 'w') as sql_file:
    # Write header
    sql_file.write("-- Database export schema\n\n")
    
    # Process each table
    for table in tables:
        print(f"Exporting schema for table: {table}")
        
        # Get table schema (column definitions)
        cursor.execute(f"""
            SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        
        # Get primary key information
        cursor.execute(f"""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = '{table}' 
            AND tc.constraint_type = 'PRIMARY KEY'
        """)
        primary_keys = [pk[0] for pk in cursor.fetchall()]
        
        # Create DROP TABLE statement
        sql_file.write(f"DROP TABLE IF EXISTS {table} CASCADE;\n")
        
        # Create CREATE TABLE statement
        sql_file.write(f"CREATE TABLE {table} (\n")
        
        # Add column definitions
        column_defs = []
        for col in columns:
            col_name, col_type, max_length, nullable, default = col
            
            # Format the column type with length if applicable
            type_str = col_type
            if max_length is not None:
                if col_type in ('character varying', 'varchar'):
                    type_str = f"{col_type}({max_length})"
            
            # Build the column definition
            col_def = f"    {col_name} {type_str}"
            
            # Add NULL/NOT NULL constraint
            if nullable == 'NO':
                col_def += " NOT NULL"
            
            # Add default value if exists
            if default is not None:
                col_def += f" DEFAULT {default}"
                
            column_defs.append(col_def)
        
        # Add primary key constraint if exists
        if primary_keys:
            pk_constraint = f"    PRIMARY KEY ({', '.join(primary_keys)})"
            column_defs.append(pk_constraint)
        
        # Join all column definitions
        sql_file.write(',\n'.join(column_defs))
        sql_file.write("\n);\n\n")
        
        # Get foreign key constraints
        cursor.execute(f"""
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}';
        """)
        
        fk_constraints = cursor.fetchall()
        for fk in fk_constraints:
            constraint_name, column_name, foreign_table, foreign_column = fk
            sql_file.write(f"ALTER TABLE {table} ADD CONSTRAINT {constraint_name} " + 
                          f"FOREIGN KEY ({column_name}) REFERENCES {foreign_table}({foreign_column});\n")
        
        # Add a separator between tables
        sql_file.write("\n")

print("Schema export completed. File saved as 'database_schema.sql'")

# Close the connection
cursor.close()
conn.close()