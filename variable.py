#variable base caller
# We have ESTIMATES of the lag and lead, but don't know it exactly, but this helps us see what of the possible 5 base combinations best fits the window
# We slide the window along the images, keeping the center base to try to determine what the original template seqeunce was. We can't cheat, the gound truth is only to
# know how long the sequene is, and later to see how accurate our base calling is.

def base_calling_uncertain_lag_lead(images, num_cycles, lag_percent, lead_percent, death_percent, num_templates, window_size):
    print ("base_calling_uncertain_lag_lead")
    base_colors = {
        'A': (255.0, 0.0, 0.0, 0.0),
        'C': (0.0, 255.0, 0.0, 0.0),
        'G': (0.0, 0.0, 255.0, 0.0),
        'T': (0.0, 0.0, 0.0, 255.0)
    }

    middle_index = window_size // 2
    called_bases = [''] * num_templates
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))

        assembled_seq = []
        for cycle in range(num_cycles - window_size + 1): # Change the loop range
            min_error = float('inf')
            best_window_seq = ""

            for base_combination in product(range(4), repeat=window_size):
                error = 0

                for w in range(window_size):
                    current_cycle = cycle + w
                    spot_color = images[current_cycle][row][col]
                    base_color = np.array(base_colors[list(base_colors.keys())[base_combination[w]]])

                    if w > 0:
                        lagging_base_color = np.array(base_colors[list(base_colors.keys())[base_combination[w - 1]]])
                    else:
                        lagging_base_color = base_color

                    if w < window_size - 1:
                        leading_base_color = np.array(base_colors[list(base_colors.keys())[base_combination[w + 1]]])
                    else:
                        leading_base_color = base_color

                    reduction_factor = 1 - np.sum(lagging_base_color * lag_percent) / base_color[np.argmax(base_color)]
                    reduction_factor -= np.sum(leading_base_color * lead_percent) / base_color[np.argmax(base_color)]

                    # Calculate the alive factor based on the death_percentage for the current cycle
                    alive_factor = (1 - death_percent) ** current_cycle

                    # Multiply the base_color by the alive_factor
                    base_color *= alive_factor

                    expected_color = base_color * reduction_factor
                    expected_color += lagging_base_color * lag_percent
                    expected_color += leading_base_color * lead_percent

                    error += np.sum((expected_color - spot_color) ** 2)

                if error < min_error:
                    min_error = error
                    best_window_seq = base_combination

            best_window_seq_letters = ''.join([list(base_colors.keys())[base_idx] for base_idx in best_window_seq])
            #print("For cycle, min_error, best_window_seq:", cycle, min_error, best_window_seq_letters)
            if cycle == 0:
                assembled_seq.extend([list(base_colors.keys())[base_idx] for base_idx in best_window_seq[:middle_index + 1]])
            else:
                assembled_seq.append(list(base_colors.keys())[best_window_seq[middle_index]])

        called_bases[seq_idx] = ''.join(assembled_seq)

    return called_bases
