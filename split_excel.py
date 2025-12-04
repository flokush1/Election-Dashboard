import pandas as pd
import sys
import os

def split_large_excel(input_file, chunk_size=50000, output_dir="split_files"):
    """
    Split a large Excel file into smaller chunks
    
    Args:
        input_file (str): Path to the large Excel file
        chunk_size (int): Number of rows per chunk (default: 50,000)
        output_dir (str): Directory to save split files
    """
    
    print(f"Processing large Excel file: {input_file}")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Read the large file
        print("Reading Excel file...")
        df = pd.read_excel(input_file)
        
        total_rows = len(df)
        print(f"Total rows: {total_rows:,}")
        
        if total_rows <= chunk_size:
            print("File is small enough, no splitting needed.")
            return
        
        # Calculate number of chunks
        num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)
        print(f"Will create {num_chunks} chunks of ~{chunk_size:,} rows each")
        
        # Get file name without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Split the file
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1:02d}_rows_{start_idx+1}_to_{end_idx}.xlsx")
            
            # Save chunk
            chunk_df.to_excel(output_file, index=False)
            
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Created: {output_file} ({len(chunk_df):,} rows, {file_size_mb:.1f}MB)")
        
        print(f"\n✅ Successfully split {input_file} into {num_chunks} files in '{output_dir}' directory")
        print(f"Each file contains ≤{chunk_size:,} rows and should be under 100MB")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_excel.py <input_file.xlsx> [chunk_size] [output_dir]")
        print("Example: python split_excel.py large_voters.xlsx 50000 split_files")
        sys.exit(1)
    
    input_file = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "split_files"
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    
    split_large_excel(input_file, chunk_size, output_dir)