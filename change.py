import json

# Function to fix the JSON format
def fix_json_file(file_name):
    with open(file_name, 'r') as f:
        content = f.read()

    # First, ensure that objects are separated by commas
    content = content.replace('}{', '}, {')

    # If the file starts or ends with invalid characters, handle it
    if not content.startswith('['):
        content = f"[{content}"
    if not content.endswith(']'):
        content = f"{content}]"

    # Attempt to load the content into a JSON object
    try:
        # Parse the corrected content
        data = json.loads(content)

        # If we get here, the content was successfully parsed
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"{file_name} fixed successfully!")
        return True
    except json.JSONDecodeError as e:
        # Handle JSON errors during decoding
        print(f"Error decoding JSON in {file_name}: {e}")
        return False

# List of files to fix
files = ['train.json', 'test.json']

# Process each file
for file_name in files:
    if not fix_json_file(file_name):
        print(f"Failed to fix {file_name}")
