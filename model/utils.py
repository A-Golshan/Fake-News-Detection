import torch
import torch_geometric as pyg
import random

def add_cls_node(data, as_node_type: str='tweet', as_edge_type: tuple=('tweet', 'reply', 'tweet')):

    num_nodes = data[as_node_type].x.size(0)

    nodes = []
    for edge_type in data.edge_types:
        
        if as_node_type not in edge_type:
            continue

        if edge_type[0] == as_node_type and edge_type[2] == as_node_type:
            nodes += data[edge_type].edge_index.flatten().tolist()
        elif edge_type[0] == as_node_type:
            nodes += data[edge_type].edge_index[0, :].tolist()
        else:
            nodes += data[edge_type].edge_index[1, :].tolist()
    nodes = set(nodes)

    new_edges = torch.tensor(
        [[node, 0] for node in nodes]
    ).t()

    data['cls'].x = torch.zeros(1, data[as_node_type].x.size(1))
    data[as_node_type, 'agg', 'cls'].edge_index = new_edges


def mask_node(
    x_dict: dict,
    masking_prob: float=0.15, 
    random_prob: float=0.1, 
    unchanged_prob: float=0.1, 
    mask_token: float=0.0
):

    # Deep copy
    masked_x_dict = {node_type: x.clone() for node_type, x in x_dict.items()}

    for node_type, x in masked_x_dict.items():

        if node_type == 'cls' or node_type == 'user':
            continue

        # Overall mask: Select 15% of tokens for prediction
        full_mask = torch.rand(x.shape[0]) < masking_prob

        # Randomly mask some of these tokens (10% of 15%)
        random_mask = full_mask & (torch.rand(x.shape[0]) < random_prob)

        # Keep some unchanged (10% of 15%)
        unchanged_mask = full_mask & (torch.rand(x.shape[0]) < unchanged_prob)

        # Mask the remaining (80% of 15%) with [MASK]
        mask_mask = full_mask & (~random_mask) & (~unchanged_mask)


        # Modify the original x tensor
        # Replace with random token
        x[random_mask] = torch.rand_like(x[random_mask])
        # Replace with [MASK]
        x[mask_mask] = mask_token
        # Unchanged_mask already preserves the original values, no action required
        
        # Update the dictionary
        masked_x_dict[node_type] = x
    
    return masked_x_dict



def within_thread_pretraining(batch):

    threads = []

    for conversation_thead in batch:

        # ramdomly sample 2 segments from the thread
        total_timesteps = conversation_thead.timesteps.item()
        # idx1, idx2 = torch.randint(0, total_timesteps, (2, ))
        t1, t2 = torch.randperm(total_timesteps)[:2] if total_timesteps > 1 else (torch.tensor(0), torch.tensor(0))

        label = 1 if t1 < t2 else 0


        data1 = conversation_thead.clone()
        data2 = conversation_thead.clone()

        for edge_type in conversation_thead.edge_types:
            mask = data1[edge_type].time <= t1
            data1[edge_type].edge_index = data1[edge_type].edge_index[:, mask]

            mask = data2[edge_type].time <= t2
            data2[edge_type].edge_index = data2[edge_type].edge_index[:, mask]
        

        data = pyg.data.Batch.from_data_list(
            [data1, data2]
        )
        data.y = torch.tensor([label])


        # Link cls node with all tweet nodes at the current step
        add_cls_node(
            data=data, 
            as_node_type='tweet', 
            as_edge_type=('tweet', 'reply', 'tweet')
        )

        threads.append(data)

    return threads

def cross_thread_pretraining(batch):

    # TODO: the number of threads is half: instead of 16 -> 8

    shuffled_batch = batch.copy()
    random.shuffle(shuffled_batch)

    threads        = []

    for i in range(0, len(shuffled_batch) - 1, 2):
        # Select two distinct threads
        data1 = shuffled_batch[i]
        data2 = shuffled_batch[i + 1]

        # Randomly sample one timestep from each thread within their time ranges
        t1 = torch.randint(0, data1.timesteps, (1, )).item()
        t2 = torch.randint(0, data2.timesteps, (1, )).item()

        for edge_type in data1.edge_types:
            mask = data1[edge_type].time <= t1
            data1[edge_type].edge_index = data1[edge_type].edge_index[:, mask]
        
        for edge_type in data2.edge_types:
            mask = data2[edge_type].time <= t2
            data2[edge_type].edge_index = data2[edge_type].edge_index[:, mask]
    
        data = pyg.data.Batch.from_data_list(
            [data1, data2]
        )
        data.y = torch.tensor([0])

        # Link cls node with all tweet nodes at the current step
        add_cls_node(
            data=data, 
            as_node_type='tweet', 
            as_edge_type=('tweet', 'reply', 'tweet')
        )
        
        threads.append(data)

    return threads

def self_supervised_phase(batch, batch_size: int=8):

    half_idx = len(batch) // 2

    wihtin_thread_batch = batch[:half_idx]
    cross_thread_batch  = batch[half_idx:]

    whithin_thread_datas = within_thread_pretraining(batch=wihtin_thread_batch)
    cross_thread_datas   = cross_thread_pretraining(batch=cross_thread_batch)

    threads = whithin_thread_datas + cross_thread_datas

    thread_loader = pyg.loader.DataLoader(
        threads, 
        batch_size=batch_size, 
        shuffle=False
    )

    return thread_loader
