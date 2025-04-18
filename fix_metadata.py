import json
import os

# Load the current metadata
with open('pdf_metadata.json', 'r') as f:
    metadata = json.load(f)

# Fix paths and check for existence
updated = False
for cid, meta in list(metadata.items()):
    # Check if the PDF exists
    if "path" in meta:
        # Clean up path format
        if meta["path"].startswith("./"):
            meta["path"] = meta["path"].replace("./", "")
            updated = True
            
        if meta["path"].startswith("/repo/"):
            meta["path"] = meta["path"].replace("/repo/", "")
            updated = True
            
        # Check if file exists at path
        if not os.path.exists(meta["path"]):
            # Try to find the file
            possible_paths = [
                os.path.join("uploaded_pdfs", f"{cid}.pdf"), 
                os.path.join("uploaded_pdfs", meta.get("filename", ""))
            ]
            
            found = False
            for alt_path in possible_paths:
                if alt_path and os.path.exists(alt_path):
                    print(f"Updating path for {cid} from {meta['path']} to {alt_path}")
                    meta["path"] = alt_path
                    found = True
                    updated = True
                    break
            
            if not found:
                print(f"Warning: PDF file not found for {cid}. Paths tried: {', '.join(possible_paths)}")
    else:
        # Try to add path
        default_path = os.path.join("uploaded_pdfs", f"{cid}.pdf")
        if os.path.exists(default_path):
            meta["path"] = default_path
            updated = True
            print(f"Added missing path for {cid}: {default_path}")

# Save the updated metadata
if updated:
    with open('pdf_metadata.json', 'w') as f:
        json.dump(metadata, f)
    print("Metadata updated and saved.")
else:
    print("No changes needed to metadata.")
