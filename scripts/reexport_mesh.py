import pytorch3d
import os
from pytorch3d.io import save_obj

path_to_data = "mesh.obj"
path_to_save = "mesh_pytorch3D.obj"


vertices, face_props, text_props = pytorch3d.io.load_obj(path_to_data)
faces = face_props.verts_idx
save_obj(path_to_save, verts=vertices, faces=faces)