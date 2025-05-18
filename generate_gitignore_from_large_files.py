import os

# max (100MB)
MAX_SIZE_BYTES = 100 * 1024 * 1024

def generate_gitignore_from_large_files(root_dir: str, output_path: str = ".gitignore", max_files: int = 50):
    large_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                size = os.path.getsize(path)
                if size >= MAX_SIZE_BYTES:
                    rel_path = os.path.relpath(path, root_dir)
                    large_files.append((rel_path.replace("\\", "/"), size))
            except Exception as e:
                print(f"Error reading {path}: {e}")

    large_files.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(large_files)} large files.")

    if not large_files:
        print("No files to ignore.")
        return

    with open(output_path, "a") as f:
        f.write("\n# Automatically added large files:\n")
        for path, size in large_files[:max_files]:
            f.write(f"{path}\n")

    print(f".gitignore updated with {min(len(large_files), max_files)} entries.")

if __name__ == "__main__":
    generate_gitignore_from_large_files(".")
