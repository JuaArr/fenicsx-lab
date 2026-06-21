from pathlib import Path

import dolfinx as dlx

def export_boundaries(export_directory: Path, mesh: dlx.mesh.Mesh, ft: dlx.mesh.MeshTags):
    with dlx.io.XDMFFile(mesh.comm, export_directory/"ft.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)