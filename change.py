import json

def fix_json_file(file_name):
    try:
        with open(file_name, 'r') as f:
            # Read the content as a single string
            content = f.read()

        # Replace '}{' with '}, {' to properly separate JSON objects
        content = content.replace('}{', '}, {')

        # Ensure the content starts with '[' and ends with ']'
        if not content.startswith('['):
            content = f"[{content}"
        if not content.endswith(']'):
            content = f"{content}]"

        # Try loading the content as JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_name}: {e}")
            return False

        # If valid, write the properly formatted content back to the file
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"{file_name} fixed successfully!")
        return True

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False

# Process your files
file_names = ['train.json', 'test.json']
for file_name in file_names:
    if not fix_json_file(file_name):
        print(f"Failed to fix {file_name}")
