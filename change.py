import json

# Function to fix the JSON format
def fix_json_file(file_name):
    with open(file_name, 'r') as f:
        content = f.read()

    # Fix the issue by adding commas between JSON objects
    content = content.replace('}{', '}, {')

    # Wrap the content in square brackets to form a valid JSON array
    content = f"[{content}]"

    # Load the corrected content into a JSON object
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_name}: {e}")
        return False

    # Write the corrected JSON data back to the file
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"{file_name} fixed successfully!")
    return True

# List of files to fix
files = ['train.json', 'test.json']

# Process each file
for file_name in files:
    if not fix_json_file(file_name):
        print(f"Failed to fix {file_name}")
