def parse_optical_file(file_path):
    # Start by reading the file
    with open(file_path, "r") as file:
        content = file.readlines()

    # Initialize the dictionary
    optical_data = {}

    # Initialize variable to keep track of the aspherical data section
    in_aspherical_section = False

    # List to temporarily store the aspherical parameters
    aspherical_parms = []

    # Variable to temporarily store the aspherical surface number
    aspherical_surface_no = None

    # Initialize a dictionary to store variable surface locations
    variable_surface_locations = {}

    # List to store variable surfaces for processing later
    variable_surfaces = []

    # Start reading from the 5th line where the actual data starts
    for line in content[:]:
        parts = line.split('\t')
        
        # Skip lines that don't contain data (these might be empty lines or headers)
        if len(parts) < 2:
            continue

        # Check if we're in the aspherical data section
        if in_aspherical_section:
            # Check if this line contains the surface number and conic constant
            if "Surface" in line:
                aspherical_surface_no = int(parts[1])
                conic = float(parts[3].replace('−', '-'))
                continue
            # Otherwise, this line should contain one of the aspherical parameters
            else:
                aspherical_parms.append(float(parts[1].replace('−', '-')))
                continue

        # Check if this line starts the aspherical data section
        if "Aspherical Data" in line:
            in_aspherical_section = True
            continue

        # Check if this line contains variable surface locations
        if parts[0].startswith("d"):
            variable_surface_no = int(parts[0][1:])  # Extract the surface number
            variable_surface_locations[variable_surface_no] = [float(p.replace('−', '-')) for p in parts[1:] if p.strip()]  # Extract the location data
            continue

        # Let's add try-except blocks to handle potential errors
        try:
            surface_no = int(parts[0].replace('*', ''))  # Keep as int, remove any '*' marking aspherical surfaces
            r = float(parts[1].replace('−', '-').replace('∞', 'inf'))  # Replace Unicode minus sign and infinity sign

            # For 'd', check if it's 'variable' and if so, set to None and store the surface number for later processing
            if parts[2].strip().lower() == '(variable)':
                d = None
                variable_surfaces.append(surface_no)
            else:
                d = float(parts[2].replace('−', '-'))  # Replace Unicode minus sign

            # For nd and νd, fill in with 1 and None respectively if they're not provided
            nd = float(parts[3].replace('−', '-')) if len(parts) > 3 else 1.0
            νd = float(parts[4].replace('−', '-')) if len(parts) > 4 else None
            diam = float(parts[5].replace('−', '-')) if len(parts) > 5 else 5.0
            # Set the type of the surface
            if '*' in parts[0]:
                surface_type = 'EVENASPH'
            else:
                surface_type = 'STANDARD'
                conic = 0.0
                aspherical_parms = [0.0]
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {e}")
            continue

        # Store in the dictionary
        optical_data[surface_no] = {'r': r, 'd': d, 'nd': nd, 'νd': νd, 'type': surface_type, 'conic': conic, 'parm': aspherical_parms, 'diam': diam}

        # Reset the aspherical data and parms for standard surfaces
        aspherical_surface_no = None
        aspherical_parms = []

    # Process variable surfaces
    for surface_no in variable_surfaces:
        optical_data[surface_no]['d'] = variable_surface_locations.get(surface_no, [None])[0]  # Use the first location value if available

    return optical_data#, variable_surface_locations


# Test the function
#optical_data, variable_surface_locations = parse_optical_file("/mnt/data/rf100300.txt")
#optical_data, variable_surface_locations

