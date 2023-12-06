from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.rna.config import RNAGraphConfig
from graphein.rna.graphs import construct_rna_graph_3d
from graphein.rna.visualisation import plotly_rna_structure_graph
from graphein.rna.edges import (
    add_all_dotbracket_edges,
    add_pseudoknots,
    add_phosphodiester_bonds,
    add_base_pairing_interactions
)

# p_config = ProteinGraphConfig()
# g_p = construct_graph(config=p_config, path='data/1ASY_r_u.pdb')

# p_1 = plotly_protein_structure_graph(
#     g_1,
#     colour_edges_by="kind",
#     colour_nodes_by="degree",
#     label_node_ids=False,
#     plot_title="Peptide backbone graph. Nodes coloured by degree.",
#     node_size_multiplier=1
#     )
# p_1.show()

# r_config = RNAGraphConfig()
# g_r = construct_rna_graph_3d(path='data/pdb_rna/1AQO.pdb')

# p_1 = plotly_rna_structure_graph(g_r)
# p_1.show()


if __name__ == '__main__':
    from RNABERT.utils.bert import Load_RNABert_Model
    model = Load_RNABert_Model('/home/steven/code/docking_classfier/RNABERT/RNABERT.pth')
    emb = model.predict_embedding('AUGC')
    print(emb.size())
