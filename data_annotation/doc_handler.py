import requests

def print_secret_message(doc_url):
    """Fetches a Google Doc with character, x, y data and prints a grid-formatted secret message."""
    if "docs.google.com/document/d/e/" in doc_url and "/pub" in doc_url:
        doc_id = doc_url.split("/d/e/")[1].split("/")[0]
        doc_url = f"https://docs.google.com/document/d/e/{doc_id}/pub?output=txt"
    elif "docs.google.com/document/d/" in doc_url:
        doc_id = doc_url.split("/d/")[1].split("/")[0]
        doc_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
    except Exception as e:
        print("Failed to fetch document:", e)
        return
    lines = response.text.strip().splitlines()
    # print("Fetched lines:")
    # for line in lines:
    #     print(repr(line))
    points = []
    max_x = max_y = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        char, x_str, y_str = parts
        # Only process lines where x_str and y_str are integers
        if not (x_str.lstrip('-').isdigit() and y_str.lstrip('-').isdigit()):
            continue
        x, y = int(x_str), int(y_str)
        points.append((x, y, char))
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    # Create grid filled with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for x, y, char in points:
        grid[y][x] = char
    for row in grid:
        print(''.join(row))
    
if __name__ == "__main__":
    print_secret_message("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")