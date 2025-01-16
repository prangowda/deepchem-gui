import os
import csv
import subprocess
from shutil import copyfile
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Constants and Configuration
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static/')
UPLOAD_DIR = os.path.join(STATIC_DIR, "data/")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print("Created data directory")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path='/static',
    template_folder=os.path.join(STATIC_DIR, 'deepchem-gui', 'templates')
)

# Routes
@app.route('/')
def webapp():
    return render_template('webapp.html')

@app.route('/upload', methods=['POST'])
def upload():
    proteins = request.files.getlist('proteins')
    ligands = request.files.getlist('ligands')
    smiles = request.files.getlist('smiles')
    smarts = request.files.getlist('smarts')

    if proteins and ligands:
        docking_result = handle_docking(proteins, ligands)
        return jsonify(docking_result)
    elif smiles:
        return jsonify(handle_smiles(smiles))
    elif smarts:
        return jsonify(handle_smarts(smarts))
    else:
        return jsonify({"error_msg": "Invalid file transfer."}), 400

# Helper Functions
def handle_docking(proteins, ligands):
    protein_paths = save_files(proteins, UPLOAD_DIR)
    ligand_paths = save_files(ligands, UPLOAD_DIR)

    docking_results = dock(protein_paths, ligand_paths)

    for i, row in enumerate(docking_results):
        for j, result in enumerate(row):
            result['protein'] = copy_and_get_url(result['protein'])
            result['ligand'] = copy_and_get_url(result['ligand'])

    return docking_results

def handle_smiles(smiles):
    smiles_path = save_file(smiles[0], UPLOAD_DIR)
    with open(smiles_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    return render_molecules(data, molecule_type="SMILES")

def handle_smarts(smarts):
    smarts_path = save_file(smarts[0], UPLOAD_DIR)
    with open(smarts_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    return render_molecules(data, molecule_type="SMARTS")

def save_files(files, directory):
    paths = []
    for file in files:
        path = save_file(file, directory)
        paths.append(path)
    return paths

def save_file(file, directory):
    path = os.path.join(directory, secure_filename(file.filename))
    file.save(path)
    return path

def copy_and_get_url(filepath):
    filename = os.path.basename(filepath)
    target_path = os.path.join(UPLOAD_DIR, filename)
    copyfile(filepath, target_path)
    return url_for('static', filename=f"data/{filename}")

def render_molecules(data, molecule_type):
    col_index = {col: i for i, col in enumerate(data[0])}
    img_dir = UPLOAD_DIR

    if molecule_type == "SMILES":
        idx = col_index["SMILES"]
    elif molecule_type == "SMARTS":
        idx = col_index["SMARTS"]
    else:
        raise ValueError(f"Unsupported molecule type: {molecule_type}")

    for i, row in enumerate(data[1:], start=1):
        try:
            molecule = Chem.MolFromSmiles(row[idx]) if molecule_type == "SMILES" else Chem.MolFromSmarts(row[idx])
            AllChem.Compute2DCoords(molecule)
            img_path = os.path.join(img_dir, f"{molecule_type.lower()}_{i}.png")
            Draw.MolToFile(molecule, img_path)
            row.append(url_for('static', filename=f"data/{os.path.basename(img_path)}"))
        except Exception as e:
            print(f"Error rendering {molecule_type} for row {i}: {e}")
            row.append("Invalid")

    return data

def dock(proteins, ligands):
    results = [[{} for _ in ligands] for _ in proteins]
    for i, protein in enumerate(proteins):
        for j, ligand in enumerate(ligands):
            docker = dc.dock.VinaGridDNNDocker(exhaustiveness=1, detect_pockets=False)
            score, (protein_docked, ligand_docked) = docker.dock(protein, ligand)
            print(f"Docking {ligand} to {protein}: score={score}")

            # Post-process docked ligand
            ligand_docked_pdb = f"{ligand_docked}.pdb"
            subprocess.run(
                ["csh", os.path.join(STATIC_DIR, 'deepchem-gui', 'scripts', 'stripqt.sh'), ligand_docked],
                check=True
            )

            results[i][j] = {
                "score": score[0],
                "protein": protein_docked,
                "ligand": ligand_docked_pdb,
            }
    return results

if __name__ == "__main__":
    app.run(debug=True)
