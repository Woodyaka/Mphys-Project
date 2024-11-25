#$!/bin/bash
#$ -cwd -V
#$ -l h_rt=1:00:00
#$ -pe smp 1
#$ -l h_vmem=16G

# Path to the Python script
SCRIPT_PATH="/home/home02/py21cb/Mphys-Project/SpE_Identification_Notebooks/SpE_Id_Alg.py"

# Check if the file exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Run the Python script
python3 "$SCRIPT_PATH"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Script executed successfully"
else
    echo "Script execution failed"
    exit 1
fi
