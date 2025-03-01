import torch
import torch_geometric as pyg

class MLP(torch.nn.Module):

    def __init__(
        self, 
        layers: list[torch.nn.Module],
        activation: torch.nn.Module=torch.nn.Sigmoid(),
        device: torch.device=torch.device('cpu')
    ):
        
        super(MLP, self).__init__()

        self.network = torch.nn.Sequential(
            *layers
        )

        self.activation = activation
        
        self.device = device
        self.to(device)
    
    def forward(self, x):

        return self.network(x)



# class GNN(torch.nn.Module):

#     def __init__(
#         self,
#         layers: list[torch.nn.Module],
#         edge_types: list,
#         aggr: str='sum',
#         TemporalEncoding: torch.nn.Module
#     ):

#         super(GNN, self).__init__()

#         self.convs = torch.nn.ModuleList(
#             [
#                 pyg.nn.HeteroConv(
#                     {
#                         edge_type: conv(
#                             in_channels,
#                             h_channels
#                         )
#                         for edge_type in edge_types
#                     },
#                     aggr=aggr
#                 )
# 		        for conv, in_channels, h_channels in layers
#             ]
#         )

#         self.TE = TemporalEncoding

#     def forward(
#         self,
#         x_dict,
#         edge_index_dict
#     ):


#         for node_type in x_dict.keys():

#             x_dict[node_type].x[:, ]
        

#         for conv in self.convs[:-1]:

#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {
#                 key: torch.nn.functional.relu(x) for key, x in x_dict.items()
#             }
#         x_dict = self.convs[-1](x_dict, edge_index_dict)
#         return x_dict


class GNN(torch.nn.Module):

    def __init__(
        self,
        layers: list[torch.nn.Module],
        edge_types: list,
        aggr: str='sum',
    ):

        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList(
            [
                pyg.nn.HeteroConv(
                    {
                        edge_type: pyg.nn.GATConv(in_channels, h_channels, add_self_loops=False) if edge_type == ('tweet', 'agg', 'cls') else conv(
                            in_channels,
                            h_channels
                        )
                        for edge_type in edge_types
                    },
                    aggr=aggr
                )
		        for conv, in_channels, h_channels in layers
            ]
        )

        # self.dropout = torch.nn.Dropout(p=0.2)

    def forward(
        self,
        x_dict,
        edge_index_dict
    ):

        for conv in self.convs[:-1]:

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: torch.nn.functional.elu(x) for key, x in x_dict.items()
            }
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict