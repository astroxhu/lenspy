def parse_optical_file(filename,radlist):
    # Open the file
    with open(filename, "r") as file:
        content = file.readlines()

    # Initialize the dictionary
    optical_data = {}
    idx=0
    # Start reading from the 5th line where the actual data starts
    for line in content[5:]:
        parts = line.split('\t')

        # Skip lines that don't contain data (these might be empty lines or headers)
        if len(parts) < 2:
            continue

        # Let's add try-except blocks to handle potential errors
        try:
            surface_no = int(parts[0])
            r = float(parts[1].replace('−', '-'))  # Replace Unicode minus sign
            d = float(parts[2].replace('−', '-'))  # Replace Unicode minus sign

            # For nd and νd, fill in with 1 and None respectively if they're not provided
            nd = float(parts[3].replace('−', '-')) if len(parts) > 3 else 1.0
            νd = float(parts[4].replace('−', '-')) if len(parts) > 4 else None
            idx+=1
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {e}")
            continue

        # Store in the dictionary
        print('idx',idx)
        idx=surface_no-1
        optical_data[surface_no] = {'r': r, 'd': d, 'nd': nd, 'νd': νd, 'type': 'STANDARD','diam': radlist[idx],'parm': [0.0], 'conic':0.0}

    return optical_data

# Use the function to parse the file
#data = parse_optical_file("rf135.txt")
#print(data)

