#!/usr/bin/env python3

with open("agents.py", "r") as f:
    agents_content = f.read()

with open("multi_agent_workflow_update.py", "r") as f:
    multi_agent_update = f.read()

with open("enhanced_multi_agent_workflow_update.py", "r") as f:
    enhanced_multi_agent_update = f.read()

# Define the start marker for multi_agent_workflow
multi_agent_start = "def multi_agent_workflow(question, tables_info, conn, db_type=\"postgresql\", model_provider=\"openai\"):"
multi_agent_end = "def enhanced_multi_agent_workflow(question, tables_info, conn, db_type=\"postgresql\", "

# Find the start position of multi_agent_workflow
multi_start_pos = agents_content.find(multi_agent_start)
if multi_start_pos == -1:
    print("Error: Could not find multi_agent_workflow function")
    exit(1)

# Find the end position of multi_agent_workflow (which is the start of enhanced_multi_agent_workflow)
multi_end_pos = agents_content.find(multi_agent_end, multi_start_pos)
if multi_end_pos == -1:
    print("Error: Could not find end of multi_agent_workflow function")
    exit(1)

# Define the start marker for enhanced_multi_agent_workflow
enhanced_multi_agent_start = "def enhanced_multi_agent_workflow(question, tables_info, conn, db_type=\"postgresql\", "

# Find a suitable end marker for enhanced_multi_agent_workflow
# Let's look for the next function definition after enhanced_multi_agent_workflow
enhanced_multi_end_marker = "\ndef "
enhanced_multi_start_pos = agents_content.find(enhanced_multi_agent_start)
if enhanced_multi_start_pos == -1:
    print("Error: Could not find enhanced_multi_agent_workflow function")
    exit(1)

# Find the end position by finding the next function definition
enhanced_multi_end_pos = agents_content.find(enhanced_multi_end_marker, enhanced_multi_start_pos + len(enhanced_multi_agent_start))
if enhanced_multi_end_pos == -1:
    # If there's no next function, we're at the end of the file
    enhanced_multi_end_pos = len(agents_content)

# Now, replace the functions in agents_content
updated_content = (agents_content[:multi_start_pos] + 
                  multi_agent_update + 
                  "\n\n" +
                  agents_content[multi_end_pos:enhanced_multi_start_pos] +
                  enhanced_multi_agent_update +
                  agents_content[enhanced_multi_end_pos:])

# Write the updated content back to agents.py
with open("agents.py", "w") as f:
    f.write(updated_content)

print("Changes applied successfully to agents.py")