import re
import sys

def fill_empty_nd_vd(input_file_path, output_file_path):
    # Constants for air
    nd_air = 1.00027717
    vd_air = 89.30
    gf_air = 0.5
    dgf_air = 0.05
    offset=-1
    withgf = False
    withdgf = False
    withdiam = 0
    # Read the input file
    with open(input_file_path, "r") as file:
        content = file.readlines()

    # Prepare the new content
    new_content = []

    # Control variable to know when to start and stop processing lines
    processing_data = False

    # Iterate over each line in the content
    for line in content:
        # If the line starts with a number (optionally followed by a '*'), it's a data line
        parts = line.lstrip().split()
        if len(parts)>1 and parts[0].strip().lower() == 'surface' and ( parts[1].strip().lower() == 'number' or parts[1].strip().lower() == 'no.'):
            print(parts)
            
            if 'θgF' in line:
                withgf=True
                idx_gf = parts.index('θgF') + offset
            if 'ΔθgF' in line:
                withdgf=True
                idx_dgf = parts.index('ΔθgF') + offset
            if 'diameter' in line:
                withdiam=1
                idx_diam = parts.index('diameter') + offset
            idx_r = parts.index('r') + offset
            idx_d = parts.index('d') + offset
            idx_nd = parts.index('nd') + offset
            if 'νd' in line:
                idx_vd = parts.index('νd') + offset
            else:
                idx_vd = parts.index('vd') + offset
              
        if re.match(r'^\d+\*?', line.strip()):
            processing_data = True

            # Split the line into parts
            #parts = line.split('\t')
            parts = line.lstrip().split()
            print (parts)
            # Check the length of parts
            if len(parts) == 3+withdiam:
                # Both nd and vd are missing, remove the newline character from the last element
                parts[-1] = parts[-1].strip()
                # Append nd and vd
                parts.insert(3,str(nd_air))
                parts.insert(4,str(vd_air))
                if withgf:
                    parts.append(str(gf_air))
                if withdgf:
                    parts.append(str(dgf_air))
                # Add the newline character back to the end of the line
                new_line = '\t'.join(parts)
            elif len(parts) == 4+withdiam:
                # Only vd is missing, remove the newline character from the last element
                parts[-1] = parts[-1].strip()
                # Append vd
                parts.insert(4,str(vd_air))
                # Add the newline character back to the end of the line
                new_line = '\t'.join(parts)
            else:
                new_line = '\t'.join(parts)
            if '\n' not in new_line:
                new_line += '\n'

            new_content.append(new_line)
        else:
            if processing_data:
                # If we were processing data and we encounter a line that doesn't start with a number, stop processing
                processing_data = False

            new_content.append(line.lstrip())

    # Write the new content to the output file
    with open(output_file_path, "w") as file:
        file.writelines(new_content)


# Fill the empty nd and vd values
name_in=sys.argv[1]+'.txt'
name_out=sys.argv[1]+'air.txt'
fill_empty_nd_vd(name_in, name_out)

