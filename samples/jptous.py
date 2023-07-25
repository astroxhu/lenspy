import sys
def convert_file_format(input_file_path, output_file_path):
    # Read the input file
    with open(input_file_path, "r") as file:
        content = file.readlines()

    # Prepare the new content
    new_content = []

    # Iterate over each line in the content
    for i, line in enumerate(content):
        # Handle the headers
        if i == 3:
            new_content.append("Surface No.\tr\td\tnd\tνd\n")
        elif i >= 4:
            # Split the line into parts
            parts = line.split()
            #print(parts)
            # Check if this is a valid data line
            if False and len(parts) < 6:
                new_content.append(line)
                continue

            # Remove the aperture data and keep the rest as it is
            new_line_parts = [p.strip() for p in parts[0:5]]
            new_line = '\t'.join(new_line_parts) + '\n'
            print(i,new_line)
            new_content.append(new_line)
        #print(new_content)

    # Write the new content to the output file
    #print(new_content)
    with open(output_file_path, "w") as file:
        file.writelines(new_content)



# Convert the file format
name_in=sys.argv[1]+'.txt'
name_out=sys.argv[1][:-2]+'.txt'
convert_file_format(name_in, name_out)
#convert_file_format("ef30028jp.txt", "ef30028.txt")
