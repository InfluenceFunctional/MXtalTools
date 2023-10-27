import torch
from ovito.data import DataCollection
from ovito.io import export_file
from ovito.pipeline import Pipeline, PythonScriptSource


def write_ovito_xyz(coords_in, atom_types_in, mol_flags_in):

    if torch.is_tensor(coords_in):
        coords_in = coords_in.cpu().detach().numpy() * 1

    if torch.is_tensor(atom_types_in):
        atom_types_in = atom_types_in.cpu().detach().numpy() * 1

    if torch.is_tensor(mol_flags_in):
        mol_flags_in = mol_flags_in.cpu().detach().numpy() * 1

    if coords_in.ndim == 3:
        coords = coords_in.reshape(coords_in.shape[0] * coords_in.shape[1], 3)
    else:
        coords = coords_in.copy()

    if atom_types_in.ndim == 2:
        atom_types = atom_types_in.reshape(atom_types_in.shape[0] * atom_types_in.shape[1])
    else:
        atom_types = atom_types_in.copy()

    if len(mol_flags_in) != len(atom_types):
        mol_flags = mol_flags_in.repeat(len(atom_types) // len(mol_flags_in))
    else:
        mol_flags = mol_flags_in.copy()

    def create_ovito_model(frame: int, data: DataCollection):
        particles = data.create_particles(count=len(coords))

        particles.create_property('Position', data=coords)
        particles.create_property('Particle Type', data=atom_types)
        particles.create_property('Mol Type', data=mol_flags)

    pipeline = Pipeline(source=PythonScriptSource(
        function=create_ovito_model))

    # Export the results of the data pipeline to an output file.
    # The system will invoke the Python function defined above once per animation frame.
    export_file(pipeline, 'traj.xyz', format='xyz',
                columns=['Position.X', 'Position.Y', 'Position.Z', 'Particle Type', 'Mol Type'],
                multiple_frames=True, start_frame=0, end_frame=0)