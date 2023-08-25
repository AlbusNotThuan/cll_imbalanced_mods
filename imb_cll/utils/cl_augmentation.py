import torch
import random
import numpy as np

def euclidean_distance(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    tensor1 = vector1.view(-1)
    tensor2 = vector2.view(-1)
    distance = (torch.norm(tensor1 - tensor2)).item()
    return distance

def cosine_similarity(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    vector1 = vector1.view(-1)
    vector2 = vector2.view(-1)

    # Normalize the vectors to unit length
    vector1 = vector1 / torch.norm(vector1)
    vector2 = vector2 / torch.norm(vector2)

    # Calculate the dot product
    dot_product = torch.dot(vector1, vector2)

    # Calculate the cosine similarity (cosine of the angle between the vectors)
    cosine_similarity = dot_product.item()

    # Calculate the cosine distance (1 minus cosine similarity)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_cl_data(x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)
            if y[i] != ytrue[j] and y[j] != ytrue[i]:
                mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
                y_a[i], y_b[i] = y[i], y[j]
                break
    return mixed_x, y_a, y_b, lam

def aug_intra_class(x, y, ytrue, k_cluster_label, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)                                                           
            if k_cluster_label[i] == k_cluster_label[j]:
                mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
                # mixed_x[i] = lam * x[i] + (1 - lam) * x[i]
                y_a[i], y_b[i] = y[i], y[j]

                # Count the violent case when true label appearing in cl label
                if y_a[i] == ytrue[j] or y_b[i] == ytrue[j]:
                    count_error += 1

                break
    return mixed_x, y_a, y_b, lam, count_error

def aug_intra_class_three_images(x, y, ytrue, k_cluster_label, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # # Generate two random mixing coefficients from a Beta distribution
        # lam1 = np.random.beta(1.0, 1.0)
        # lam2 = np.random.beta(1.0, 1.0)

        # # Calculate the third mixing coefficient to satisfy the sum constraint
        # lam3 = 1 - lam1 - lam2
        # lam3 = max(0, lam3)  # Ensure lam3 is non-negative

        # # Normalize the coefficients to ensure they sum up to 1
        # total_lam = lam1 + lam2 + lam3
        # lam1 /= total_lam
        # lam2 /= total_lam
        # lam3 /= total_lam
        lam1 = 0.34
        lam2 = 0.33
        lam3 = 0.33
    else:
        lam1 = 0.34
        lam2 = 0.33
        lam3 = 0.33
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b, y_c = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)                                                           
            if k_cluster_label[i] == k_cluster_label[j]:
                while(True):
                    k = random.randint(0, batch_size -1)
                    if k_cluster_label[i] == k_cluster_label[k]:
                        mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k]
                        y_a[i], y_b[i], y_c[i] = y[i], y[j], y[k]
                        # Count the violent case when true label appears in cl label
                        if (
                            y[i] == ytrue[j]
                            or y[i] == ytrue[k] 
                            or y[j] == ytrue[i] 
                            or y[j] == ytrue[k] 
                            or y[k] == ytrue[i] 
                            or y[k] == ytrue[j]
                        ):
                            count_error += 1

                        break
                break

    # for i in range(batch_size):
    #     # Gather indices of samples in the same cluster as i
    #     same_cluster_indices = [j for j in range(batch_size) if k_cluster_label[i] == k_cluster_label[j]]
        
    #     j = random.choice(same_cluster_indices)
    #     # Remove j from the indices to prevent selecting it again
    #     same_cluster_indices.remove(j)

    #     # Randomly select another sample from the remaining same-cluster indices
    #     k = random.choice(same_cluster_indices)
        
    #     mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k]
    #     y_a[i], y_b[i], y_c[i] = y[i], y[j], y[k]

    #     # Count the violent case when true label appears in cl label
    #     if (
    #         y_a[i] == ytrue[j] 
    #         or y_b[i] == ytrue[j] 
    #         or y_c[i] == ytrue[j] 
    #         or y_a[i] == ytrue[k] 
    #         or y_b[i] == ytrue[k] 
    #         or y_c[i] == ytrue[k]
    #     ):
    #         count_error += 1
    return mixed_x, y_a, y_b, y_c, lam1, lam2, lam3, count_error

def aug_intra_class_four_images(x, y, ytrue, k_cluster_label, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # # Generate three random mixing coefficients from a Beta distribution
        # lam1 = np.random.beta(1.0, 1.0)
        # lam2 = np.random.beta(1.0, 1.0)
        # lam3 = np.random.beta(1.0, 1.0)
        # # Calculate the fourth mixing coefficient to satisfy the sum constraint
        # lam4 = 1 - lam1 - lam2 - lam3
        # lam4 = max(0, lam4)  # Ensure lam4 is non-negative
        
        # # Normalize the coefficients to ensure they sum up to 1
        # total_lam = lam1 + lam2 + lam3 + lam4
        # lam1 /= total_lam
        # lam2 /= total_lam
        # lam3 /= total_lam
        # lam4 /= total_lam
        lam1 = 0.25
        lam2 = 0.25
        lam3 = 0.25
        lam4 = 0.25
    else:
        lam1 = 0.25
        lam2 = 0.25
        lam3 = 0.25
        lam4 = 0.25
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b, y_c, y_d = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)                                                           
            if k_cluster_label[i] == k_cluster_label[j]:
                while(True):
                    k = random.randint(0, batch_size -1)
                    if k_cluster_label[i] == k_cluster_label[k]:
                        while(True):
                            l = random.randint(0, batch_size -1)
                            if k_cluster_label[i] == k_cluster_label[l]:
                                mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k] + lam4 * x[l]
                                y_a[i], y_b[i], y_c[i], y_d[i] = y[i], y[j], y[k], y[l]
                                # Count the violent case when true label appears in cl label
                                if (
                                    y[i] == ytrue[j]
                                    or y[i] == ytrue[k] 
                                    or y[i] == ytrue[l] 
                                    or y[j] == ytrue[i] 
                                    or y[j] == ytrue[k]
                                    or y[j] == ytrue[l]  
                                    or y[k] == ytrue[i] 
                                    or y[k] == ytrue[j]
                                    or y[k] == ytrue[l]
                                    or y[l] == ytrue[i] 
                                    or y[l] == ytrue[j]
                                    or y[l] == ytrue[k]
                                ):
                                    count_error += 1
                                break

                        break
                break

    return mixed_x, y_a, y_b, y_c, y_d, lam1, lam2, lam3, lam4, count_error

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.squeeze()
    y_b = y_b.squeeze()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_k(n1, n2, f):
    k1 = pow(n1, f)
    k2 = pow(n2, f)
    return k1, k2


def get_lambda(x, k1, k2):
    lambda_lower = 0.0
    t_lower = 1.0
    lambda_upper = 1.0
    t_upper = 0.0
    lambda_middle = k1 / (k1 + k2)
    t_middle = 0.5
    if x < lambda_middle:
        lambda_target = ((-t_middle) *
                         (x - lambda_lower) / lambda_middle) + t_lower
    elif x > lambda_middle:
        lambda_target = ((x - lambda_upper) * (t_middle - t_upper) /
                         (lambda_middle - lambda_upper))
    else:
        raise ValueError("[-] Check Boundary Case !")
    return lambda_target

def mamix_intra_aug(x, y, k_cluster_label, mamix_ratio, cls_num_list, device, alpha=1.0):
    if alpha > 0:
        lam_x = np.random.beta(alpha, alpha)
    else:
        lam_x = 1

    cls_num_list = torch.tensor(cls_num_list)

    batch_size = x.size()[0]
    # get the index from random permutation for mix x
    # index = torch.randperm(batch_size)

    # check will store the pair chosen for mixup with each other [batch, 2]
    # check = []
    # for i, j in enumerate(index):
    #     check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item()])
    # check = torch.tensor(check)

    check = []
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)                                                         
            if k_cluster_label[i] == k_cluster_label[j]:
                check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item(), j])
                break
    check = torch.tensor(check)

    # Now, we are going to compute lam_y for every pair
    lam_y = list()
    new_index = list()
    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0].item()
        temp2 = check[i][1].item()
        new_index.append(check[i][2].item())

        f = mamix_ratio
        k1, k2 = get_k(temp1, temp2, f)
        lam_t = get_lambda(lam_x, k1, k2)

        lam_y.append(lam_t)

    lam_y = torch.tensor(lam_y).to(device)

    mixed_x = (1 - lam_x) * x + lam_x * x[new_index, :]
    y_a, y_b = y, y[new_index]

    return mixed_x, y_a, y_b, lam_x, lam_y

def mamix_criterion(criterion, pred, y_a, y_b, lam_y, args):
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(
        criterion(pred, y_b), (1 - lam_y))

    return loss.mean()