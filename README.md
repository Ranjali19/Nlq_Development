# Project Overview

This project is designed to facilitate the generation of SQL queries from natural language text using a LangGraph-based agent. It integrates with a SQLite database and allows users to upload Excel files, which are then processed to create SQL queries based on user input.

## File Structure

- `src/update_app.py`: Contains the main application logic for processing Excel files, generating SQL queries, and interacting with a SQLite database. It includes functions for uploading data, generating table summaries, and executing SQL queries based on user input.

- `src/graph.py`: Implements a LangGraph-based agent for generating SQL queries from text. It defines a workflow using the `StateGraph` class, adding nodes for various SQL-related tasks such as `sql_autoquery`, `get_table_heads`, and `get_extended_table_description`. The agent also includes routing and evaluation logic to determine the appropriate actions based on user queries. The `visualize_agent` function provides a way to visualize the agent's workflow.

- `requirements.txt`: Lists the dependencies required for the project, including libraries for LangGraph and any other necessary packages.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application by executing the following command:

   ```
   python src/update_app.py
   ```

2. Upload Excel files through the web interface.
3. Ask questions related to the data, and the application will generate SQL queries to retrieve the relevant information.

## Features

- Upload and process multiple Excel files.
- Generate structured summaries of tables.
- Execute SQL queries based on user input.
- Visualize the workflow of the LangGraph-based agent for SQL query generation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.