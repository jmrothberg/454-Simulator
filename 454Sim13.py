#454 Sequencee simulator with strand simulator for text visualization and dark bases
#Based on Sequencing by synthesis (Invented by Jonathan Rothberg) with 454 E-wave technology and Lightning terminators.
#Jonathan Rothberg, March 2023
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import itertools


# Define the complement dictionary
complement_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N': 'N', '-': '-'}
dye_dict = {'A': 'yellow', 'C': 'green', 'G': 'blue', 'T': 'red'}

# Increase if you have faster computer than Raspberry Pi
num_strands = 1000

tauUV = input("Enter tauUV (default 0.3): ")
if tauUV:
    tauUV = float(tauUV)
else:
    tauUV = 0.3
uv_time = input("Enter uv_time (default 1): ")
if uv_time:
    uv_time = float(uv_time)
else:
    uv_time = 1
tauEX = input("Enter tauEX (defualt 10): ")
if tauEX:
    tauEX = float(tauEX)
else:
    tauEX = 10
ex_time = input("Enter ex_time (defualt 100): ")
if ex_time:
    ex_time = float(ex_time)
else:
    ex_time = 100

p_die = input("Enter p_die percent per second of UV each cycle (increases concentration of dark bases, default 0.02): ")
if p_die:
    p_die = float(p_die)
else:
    p_die = .02

p_dark = input("Enter p_dark percent chance per second of UV that a strand dyes in a cyle (kills a strand default 0.01): ")
if p_dark:
    p_dark = float(p_dark)
else:
    p_dark = .01

template = input ("Enter DNA Sequence (default GATCGATCG....) ")
if not template:
    template = 'GATCGATCGACCGTAGCTAGGCGATCGAGCGTGAACTAACAATTCCGGTACATGACGTACCTTGGCATCGATCGGGTAGAGCGTGAACTAACGTACGTACGTACCTTGGCATCGATCGGGTACCTTACGGTTAAAACTCGATCACGTAGCTAGCGATCGG'
template = template.upper()
#To make sure you don't run off and give errors
template = template + "A" * 20

cycle_count = input("Enter cycle_count (default 20): ")
if cycle_count:
    cycle_count = int(cycle_count)
    if cycle_count > len(template):
        cycle_count = len(template)
        print("Cycle count was reset to the length of the template.")
else:
    cycle_count= 20

if cycle_count > len(template):
    cycle_count = len(template)
    print("Cycle count was reset to the length of the template.")

# Initialize the strands
strands = []
for i in range(num_strands):
   strands.append([' ',' '])

# Define the colors for plotting
colors = ['yellow', 'green', 'blue', 'red']

# Define a function to simulate a cycle of sequencing
def simulate_cycle(strands, template, cycle):
    # Loop over the strands
    for i, strand in enumerate(strands):
        # Skip the iteration if the strand is already longer than the template
        if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
            continue
        # Check if the strand unblocks during the UV window and extends and then unblocks again
        # Generate the random number for the first unblocking event
        uv_extend_count = 0
        rand_num1 = np.random.random()
        if strand[-1] in ['yellow', 'green', 'blue', 'red'] and rand_num1 < (1 - np.exp(-uv_time / tauUV)):
            # Calculate the time elapsed in the UV window for the first event
            time_elapsed1 = -tauUV * np.log(1 - rand_num1)
            # Subtract the time elapsed from the original uv_time
            nuv_time = uv_time - time_elapsed1
            # Remove the dye label
            strand = strand[:-1]
            uv_unblock_count = 1
            # Loop to add bases during the UV window which if cleaved after addition are invisible they move you forward in a frame!
            while nuv_time > 0:
                rand_num2 = np.random.random()
                #tauEX Needs to know the next base in template so it can get tauEX(base)
                #
                if strand[-1] not in ['yellow', 'green', 'blue', 'red'] and rand_num2 < (1 - np.exp(-nuv_time / tauEX)):
                    uv_extend_count = uv_extend_count + 1
                    time_elapsed2 = -tauEX * np.log(1 - rand_num2)
                    nuv_time = nuv_time - time_elapsed2
                    next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                    complement_base = complement_dict[next_base_in_template]
                    dye = dye_dict[complement_base]
                    if strand[-1].islower() and strand[-2].islower():
                        strand += [complement_base, dye]
                    elif strand[-1].islower():
                        strand += [' ', complement_base, dye]
                    else:
                        strand += [' ', ' ', complement_base, dye]
                    rand_num3 = np.random.random()
                    if rand_num3 < (1 - np.exp(-nuv_time / tauUV)):
                        uv_unblock_count = uv_unblock_count + 1
                        time_elapsed3 = -tauUV * np.log(1 - rand_num3)
                        nuv_time = nuv_time -  time_elapsed3
                        last_element = strand[-2].lower()
                        #print("uv_unblock_count, Removed: ", uv_unblock_count, strand[-(6-uv_unblock_count):])
                        if uv_unblock_count > 4:
                            uv_unblock_count = 4
                        strand[-(6-uv_unblock_count):] = [last_element]
                        #print("Replaced with: ", [last_element])
                else:
                    # Break out of the loop if the if statement is not true
                    break

        extension_count = uv_extend_count

        # UV Killing of strand. Updated to be function of uv_time
        if strand[-1] != 'dead' and np.random.random() < p_die * uv_time:
            strand.append('dead')

        # Check if the strand extends during the second part of the cycle
        dark_count = 0
        rand_num1 = np.random.random()
        # Do you extend in the ex_time window
        if strand[-1] not in ['dead', 'yellow', 'green', 'blue', 'red'] and rand_num1 < (1 - np.exp(-ex_time / tauEX)):
            time_elapsed1 = -tauEX * np.log(1 - rand_num1)
            nex_time = ex_time - time_elapsed1
            percent_blocked = (1 - p_dark * uv_time) ** cycle # amount of labeled and blocked nucleotides
            rand_numDark = np.random.random()
            if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
                    break
            # Extend the strand and add a new dye - This is NORMAL behavior,
            if rand_numDark < percent_blocked:   # Do you put on a good base with dye e.g. blocked.
                next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                complement_base = complement_dict[next_base_in_template]
                dye = dye_dict[complement_base]
                if strand[-1].islower() and strand[-2].islower():
                    strand += [complement_base, dye]
                elif strand[-1].islower():
                    strand += [' ', complement_base, dye]
                else:
                    strand += [' ', ' ', complement_base, dye]
                extension_count += 1
            else:
                #Dark base loop. N in the print for first dark base so we know keeps frame.  Assume unblocked and dark.
                #Need to check time remaining before each extension
                #print ("Added first dark base, cycle, percent unblocked, nex_time :", cycle , percent_unblocked, nex_time)
                next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                complement_base = 'N' # it is dark so you don't know what is added :)
                #Spacing so if lower case based before you need to move over less ot make the base align with the other bases in print outs
                #all 3 of these conditions occur here and in two places above
                if strand[-1].islower() and strand[-2].islower():
                    strand += [complement_base]
                elif strand[-1].islower():
                    strand += [' ', complement_base]
                else:
                    strand += [' ', ' ', complement_base]
                dark_count = dark_count + 1
                while nex_time > 0:
                    if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
                        break
                    rand_num2 = np.random.random()
                    #Use remaining time nex_time to adjust probabilities
                    if strand[-1] not in ['dead', 'yellow', 'green', 'blue', 'red'] and rand_num2 < (1 - np.exp(-nex_time / tauEX)):
                        time_elapsed2 = -tauEX * np.log(1 - rand_num2) # how long did it take
                        nex_time = nex_time - time_elapsed2 # lose the time in the extension phase
                        rand_numDark = np.random.random()
                        #if you extend do you get a dark base?
                        if rand_numDark > percent_blocked:
                            #print ("Added addional out of phase dark bases in cycle, percent unblocked, nex_time :", cycle , percent_unblocked, nex_time)
                            next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                            complement_base = 'n' # it is dark so you don't know what is added :)
                            strand += [complement_base]
                            dark_count = dark_count +1
                        else:
                            break
                    else:
                        # Break out of the loop if the if statement is not true
                        break

        #This makes a strand LAG because no bases where added in cycle
        #Insert the spaces before the dye which stays at end
        if strand[-1] != 'dead' and extension_count == 0 and dark_count == 0:
            if strand[-1] in ['yellow', 'green', 'blue', 'red']:
                strand.insert(-1, " ")
                strand.insert(-1, " ")
                strand.insert(-1, ".")
            else:
                strand.append(" ")
                strand.append(" ")
                strand.append(".")

        # Update the original strands list with the modified strand from the for loop
        strands[i] = strand
    return

# Initialize the dye counts
dye_counts = np.zeros((cycle_count, 4))

# Loop over the cycles and simulate sequencing
for cycle in range(cycle_count):
    simulate_cycle(strands, template, cycle)
    # Count the number of each dye in strands
    for strand in strands:
        if 'yellow' in strand:
            dye_counts[cycle][0] += 1
        if 'green' in strand:
            dye_counts[cycle][1] += 1
        if 'blue' in strand:
            dye_counts[cycle][2] += 1
        if 'red' in strand:
            dye_counts[cycle][3] += 1

print (dye_counts)

# Create complement of the template by using a list comprehension and the complement_dict
complement_template = [complement_dict[base] for base in template[:cycle_count]]
# Convert the list of complements back into a string
complement_string = ''.join(complement_template)

print ("Template: \n", template)

# Base calling code
threshold = 0.2 * np.amax(dye_counts, axis=1)
base_calls = []
for i in range(cycle_count):
    max_count = np.amax(dye_counts[i])
    max_index = np.argmax(dye_counts[i])
    second_max_count = np.partition(dye_counts[i], -2)[-2]
    if max_count >= (1 + 0.2) * second_max_count:
        base_call = ['A', 'C', 'G', 'T'][max_index]
    else:
        base_call = "N"
    base_calls.append(base_call)
base_call_string = "".join(base_calls)

print("Single cycle calls: \n", base_call_string)
print("Complement: \n", complement_string)

# Set the figure width and height
fig_width = min(4 + cycle_count / 10, 16)
fig_height = 5

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
max_height = np.sum(dye_counts, axis=1).max()
ax.set_ylim(0, 1.2 * max_height)

# Plot the stacked bar graph
ax.bar(np.arange(cycle_count), np.sum(dye_counts, axis=1), color='gray', edgecolor='black')
ax.bar(np.arange(cycle_count), dye_counts[:, 0], color='yellow', bottom=np.sum(dye_counts[:, 1:], axis=1), label='A')
ax.bar(np.arange(cycle_count), dye_counts[:, 1], color='green', bottom=np.sum(dye_counts[:, 2:], axis=1), label='C')
ax.bar(np.arange(cycle_count), dye_counts[:, 2], color='blue', bottom=dye_counts[:, 3], label='G')
ax.bar(np.arange(cycle_count), dye_counts[:, 3], color='red', label='T')

# Add the base calls
for i in range(cycle_count):
    ax.text(i, max_height * 1.1, base_calls[i], ha='center', va='bottom')
    ax.text(i, max_height, complement_template[i], ha='center', va='bottom', fontsize=10)
    ax.text(i, 0, template[i], ha='center', va='bottom', fontsize=10)

# Set the x-axis ticks and labels to integers
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))


# Add the labels
ax.legend()
ax.set_xlabel('Cycle')
ax.set_ylabel('Counts')
ax.set_title('Simulated Sequencing Counts')

# Add the variable values
text = "454 Sim JMR \n tauUV: {}\ntauEX: {}\nuv_time: {}\nex_time: {}\nnum_strands: {}\ncycle_count: {}\np_die: {}\n p_dark: {}".format(
    tauUV, tauEX, uv_time, ex_time, num_strands, cycle_count, p_die, p_dark)
ax.annotate(text, xy=(1, 1), xycoords='figure fraction', ha='right', va='top', fontsize=8)

# Save to a temporary file
temp_file = 'temp.png'
plt.savefig(temp_file, bbox_inches='tight')
plt.show()

#Strand length not counting spaces or periods or dye
strand_lengths = [len([x for x in strand if x != ' ' and x != '.' and len(x) == 1]) for strand in strands]
#Perfect reads in phase, which means need to terminate if you hit a . or a N or a n - need to rewrite this.

strand_lengths_in_phase = []
for strand in strands:
    length = 0
    for character in strand:
        if character not in ['.', 'n', ' '] and character.isupper():        #I removed 'N' so you are allowed an in frame N
            length += 1                                                     #Can make logic more complex if you want to allow a .n to be in frame
        elif character == ' ':
            pass
        else:
            break
    strand_lengths_in_phase.append(length)

phase_counts = dict(Counter(strand_lengths_in_phase))
xp = list(phase_counts.keys())
yp = list(phase_counts.values())

length_counts = dict(Counter(strand_lengths))
x = list(length_counts.keys())
y = list(length_counts.values())

plt.bar(x, y, color='blue', alpha=0.5, label='All Strands')
plt.bar(xp, yp, color='red', alpha=0.5, label='In Phase')
plt.xlabel("Strand Length")
plt.ylabel("Number of Strands")
plt.title("Number of Strands at Each Strand Length")
plt.legend()
temp_dist = 'temp_dist.png'
plt.savefig(temp_dist, bbox_inches='tight')
plt.show()

filename = input("Enter a file name to save the data (or press Enter to skip): ")

if filename:
    dye_counts_file = filename + ".txt"
    figure_file = filename + "_seq.png"
    dist_file = filename + "_dist.png"
    with open(dye_counts_file, 'w') as f:
        f.write("#454 Sequence simulator with strand simulator for text visualization\n")
        f.write("#Jonathan Rothberg, March 2023\n")
        f.write("#tauUV: {}\n".format(tauUV))
        f.write("#uv_time: {}\n".format(uv_time))
        f.write("#tauEX: {}\n".format(tauEX))
        f.write("#ex_time: {}\n".format(ex_time))
        f.write("#p_die: {}\n".format(p_die))
        f.write("#p_dark: {}\n".format(p_dark))
        f.write("#num_strands: {}\n".format(num_strands))
        f.write("#cycle_count: {}\n".format(cycle_count))
        f.write("#template: {}\n".format(template))
        f.write("#complement_string: {}\n".format(complement_string))
        f.write("#base_call_string: {}\n".format(base_call_string))
        np.savetxt(f, dye_counts, fmt ='%d', delimiter=",")
    os.rename(temp_file, figure_file)
    os.rename(temp_dist, dist_file)
else:
    # If the user doesn't want to save the plot, delete the temporary files
    os.remove(temp_file)
    os.remove(temp_dist)
for strand in strands:
    print("".join(strand))

save_strands = input("Y or any key to save (or hit return to skip): ")
if save_strands:
    strands_file = filename + "_strands.text"
    print ("saving ", strands_file)
    with open(strands_file, "w") as f:
        for strand in strands:
            f.write("".join(strand) + "\n")
