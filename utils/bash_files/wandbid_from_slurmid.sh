#!/bin/bash

# Define the function
extract_wandbid_id() {
    local outp_path="$HOME/slurmfiles/"
    local outp_file_path=$(ls -ltr "${outp_path}"/*${SLURM_JOB_ID}* 2>/dev/null | tail -n 1 | awk '{print $NF}')
    local id_str_prefix="Creating sweep with ID: "

    # Initialize a wandbid to keep track of whether the ID has been found
    found=0

    # Loop until the wandbid is found
    while [ $found -eq 0 ]; do
        # Check if the file contains the desired string and extract the wandbid
        if grep -q "$id_str_prefix" "$outp_file_path"; then
            # Extract the wandbid following "Creating sweep with ID: "
            local wandbid=$(grep "$id_str_prefix" "$outp_file_path" | head -n 1 | sed "s/.*$id_str_prefix//")

            # Mark as found
            found=1

            # WARNING: bash returns the variable using "echo" so don't return any other
            echo $wandbid
            break

        else
            # Wait for 1 second before retrying
            sleep 1
        fi
    done
}