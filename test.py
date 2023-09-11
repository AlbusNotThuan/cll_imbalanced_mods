def aug_intra_class_three_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # Generate three random mixing coefficients from a Beta distribution
        lam1 = np.random.beta(alpha, alpha)
        lam2 = np.random.beta(alpha, alpha)
        # Calculate the fourth mixing coefficient to satisfy the sum constraint
        lam3 = 1 - lam1 - lam2
        lam3 = max(0, lam3)  # Ensure lam3 is non-negative
        # Normalize the coefficients to ensure they sum up to 1
        total_lam = lam1 + lam2 + lam3
        lam1 /= total_lam
        lam2 /= total_lam
        lam3 /= total_lam
    else:
        lam1 = 1/3
        lam2 = 1/3
        lam3 = 1/3
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b, y_c = torch.zeros_like(x).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    # Precompute random indices that satisfy the condition
    matching_indices = (torch.tensor(k_cluster_label)[:, None] == torch.tensor(k_cluster_label)).clone().detach()
    lambda_y = []
    label_y = []

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2))
            j, k, l, n, m, q, r, u, v = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 9)).clone().detach()

            # Define the indices and initialize the lam values
            indices = [i, j, k, l, n, m, q, r, u, v]
            lam_values = [0] * 10
            y_values = [0] * 10

            # Calculate distances and lam values in a loop
            distances = [euclidean_distance(x[i], x[index]) if euclidean_distance(x[i], x[index]) != 0 else 10 for index in indices]

            # Calculate lam values
            for idx, distance in enumerate(distances):
                lam_values[idx] = (1 / distance) / sum(1 / dist for dist in distances)
            lam_values = torch.tensor(lam_values)

            # Now, lam_values contains the lam values for each index
            # lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = lam_values

            # # Move tensors to CPU if they are on CUDA
            # mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
            # y_a[i], y_b[i], y_c[i] = y[i], y[j], y[k]

            # Perform weighted calculations for mixed_x
            if dataset_name in ("CIFAR10", "CIFAR20"):
                mixed_x[i] = x[i] #sum(lam * x[idx] for lam, idx in zip(lam_values, indices))
            else:
                mixed_x[i] = x[i]

            y_values = torch.tensor([y[index] for index in indices])

            lam_values, y_values = recalculate_lambda_label_sharing(y_values, lam_values)

            lambda_y.append(lam_values)
            label_y.append(y_values)

            # Count the violent case when true label appears in cl label
            # if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[k] == ytrue[i] or y[k] == ytrue[j]):
            #     count_error += 1

return mixed_x, label_y, lambda_y

def recalculate_lambda_label_sharing(y_values, lam_values):
    final_lam = []
    final_y_values = []
    for i in range(len(y_values)):
        i_value = 0
        for j, l in zip(y_values, lam_values):
            if i == j:
                i_value = i_value + l
        final_lam.append(i_value)

    for y, index in zip(range(len(y_values)), final_lam):
        if index == 0:
            new_y = 0
        else:
            new_y = y
        final_y_values.append(new_y)
    final_lam = torch.tensor(final_lam)
    final_y_values = torch.tensor(final_y_values)
    
    return final_lam, final_y_values