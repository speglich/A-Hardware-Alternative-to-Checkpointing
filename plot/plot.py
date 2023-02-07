import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tikzplotlib
import os

from matplotlib.pyplot import figure

#------------------------------------------------- #
#------------------------------------------------- #
#------------------------------------------------- #
# Main Parameters
#------------------------------------------------- #
#------------------------------------------------- #
#------------------------------------------------- #

title = "1 Socket - 26 Physical Cores"
png_folder = "figures"
tex_folder = "figures-tex"
x_label = "Number of disks"

#title = "MPI - 2 Sockets - 52 Physical Cores"
#png_folder = "mpi-figures"
#tex_folder = "mpi-figures-tex"
#x_label = "Number of disks per socket"

if not os.path.exists(png_folder):
    os.makedirs(png_folder)

if not os.path.exists(tex_folder):
    os.makedirs(tex_folder)

#------------------------------------------------- #
#------------------------------------------------- #
# Forward Propagation
#------------------------------------------------- #
#------------------------------------------------- #

operator = "Forward Propagation"

#------------------------------------------------- #
# Data Manipulation
#------------------------------------------------- #

fwd = pd.read_csv('forward.csv')
fwd = fwd.sort_values(by=['Disks'])
fwd.set_index('Disks', inplace=True)

fwd["Forward Propagation Time"] = fwd[' [FWD] Section0']  + fwd[' [FWD] Section1'] + fwd[' [FWD] Section2']
fwd["Write Time"] = fwd[' [IO] Open'] + fwd[' [IO] Write'] + fwd[' [IO] Close']
fwd['Ratio'] = fwd["Write Time"] / fwd["Forward Propagation Time"]

fwd['GB'] = fwd[' Bytes'] / 1000000000
fwd['Write Troughput'] = fwd['GB'] / fwd [' [IO] Write']

labels = list(fwd.index.values)
fwd['Write Troughput per disk' ] = fwd['Write Troughput'] / labels

#------------------------------------------------- #
# Execution time
#------------------------------------------------- #
experiment = "Execution Time [s]"
output = "forward-exec-time"
#------------------------------------------------- #

exec_time = fwd[["Forward Propagation Time", "Write Time"]]

ram_time = exec_time._get_value(0, "Forward Propagation Time")

exec_time = exec_time.drop(index=(0))

ax = exec_time.plot.bar(stacked=True, grid=True)

ax.axhline(y=ram_time, color='r', linestyle='--', label="RAM Execution Time")

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(exec_time.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Write time
#------------------------------------------------- #
experiment = "Write Time [s]"
output = "forward-write-time"
#------------------------------------------------- #

write_time = fwd[['Write Time']]

write_time = write_time.drop(index=(0))

ax = write_time.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(write_time.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Throughput
#------------------------------------------------- #
experiment = "Write Troughput [GB/s]"
output = "forward-gbps-write"
#------------------------------------------------- #

throughput = fwd[['Write Troughput']]
throughput = throughput.drop(index=(0))

ax = throughput.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Throughput [GB/s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Throughput per NVME
#------------------------------------------------- #
experiment = "Write Troughput per disk [GB/s]"
output = "forward-gbps-write-per-nvme"
#------------------------------------------------- #

throughput = fwd[['Write Troughput per disk']]
throughput = throughput.drop(index=(0))

ax = throughput.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Throughput [GB/s] per disk')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Write / Compute Ratio
#------------------------------------------------- #
experiment = "Write Time / Compute Time Ratio"
output = "forward-ratio"
#------------------------------------------------- #

total = fwd[['Ratio']]
total = total.drop(index=(0))

ax = total.plot.bar(stacked=True, grid=True)

ax.axhline(y=1, color='r', linestyle='--', label="Write equals to compute")

ax.set_ylabel('Ratio')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
#------------------------------------------------- #
# Adjoint Calculation
#------------------------------------------------- #
#------------------------------------------------- #

operator = "Adjoint Calculation"

#------------------------------------------------- #
# Data Manipulation
#------------------------------------------------- #

rev = pd.read_csv('reverse.csv')
rev = rev.sort_values(by=['Disks'])
rev.set_index('Disks', inplace=True)

rev["Adjoint Calculation Time"] = rev[' [REV] Section0']  + rev[' [REV] Section1'] + rev[' [REV] Section2']
rev["Read Time"] = rev[' [IO] Open'] + rev[' [IO] Read'] + rev[' [IO] Close']

rev['GB'] = rev[' Bytes'] / 1000000000
rev['Read Troughput'] = rev['GB'] / rev [' [IO] Read']

labels = list(rev.index.values)
rev['Read Troughput per disk' ] = rev['Read Troughput'] / labels

#------------------------------------------------- #
# Execution time
#------------------------------------------------- #
experiment = "Execution Time [s]"
output = "reverse-exec-time"
#------------------------------------------------- #

exec_time = rev[["Adjoint Calculation Time", "Read Time"]]

ram_time = exec_time._get_value(0, "Adjoint Calculation Time")

exec_time = exec_time.drop(index=(0))

ax = exec_time.plot.bar(stacked=True, grid=True)

ax.axhline(y=ram_time, color='r', linestyle='--', label="RAM Execution Time")

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(exec_time.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Read time
#------------------------------------------------- #
experiment = "Read Time [s]"
output = "reverse-read-time"
#------------------------------------------------- #

write_time = rev[['Read Time']]

write_time = write_time.drop(index=(0))

ax = write_time.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(write_time.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Throughput
#------------------------------------------------- #
experiment = "Read Troughput [GB/s]"
output = "reverse-gbps-read"
#------------------------------------------------- #

throughput = rev[['Read Troughput']]
throughput = throughput.drop(index=(0))

ax = throughput.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Throughput [GB/s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Throughput per NVME
#------------------------------------------------- #
experiment = "Read Troughput per disk [GB/s]"
output = "reverse-gbps-read-per-nvme"
#------------------------------------------------- #

throughput = rev[['Read Troughput per disk']]
throughput = throughput.drop(index=(0))

ax = throughput.plot.bar(stacked=True, grid=True)

ax.set_ylabel('Throughput [GB/s] per disk')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
#------------------------------------------------- #
# Total
#------------------------------------------------- #
#------------------------------------------------- #

operator = "Total"

#------------------------------------------------- #
# Data Manipulation
#------------------------------------------------- #

#------------------------------------------------- #
# Execution time
#------------------------------------------------- #
experiment = "Execution Time [s]"
output = "total-exec-time"
#------------------------------------------------- #

fwd_exec_time = fwd [["Forward Propagation Time", "Write Time"]]
rev_exec_time = rev [["Adjoint Calculation Time", "Read Time"]]

exec_time = pd.concat([fwd_exec_time, rev_exec_time], axis=1)

exec_time = exec_time[["Forward Propagation Time", "Adjoint Calculation Time", "Write Time", "Read Time"]]

ram_time = exec_time._get_value(0, "Adjoint Calculation Time") + exec_time._get_value(0, "Forward Propagation Time")

exec_time = exec_time.drop(index=(0))

ax = exec_time.plot.bar(stacked=True, grid=True)

ax.axhline(y=ram_time, color='r', linestyle='--', label="RAM Execution Time")

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(exec_time.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)

#------------------------------------------------- #
# Slowdown
#------------------------------------------------- #
experiment = "Performance Impact"
output = "total-slowdown"
#------------------------------------------------- #

exec_time = pd.concat([fwd_exec_time, rev_exec_time], axis=1)

exec_time = exec_time[["Forward Propagation Time", "Adjoint Calculation Time", "Write Time", "Read Time"]]

exec_time["Total"] = exec_time["Forward Propagation Time"] + exec_time["Adjoint Calculation Time"] \
                + exec_time["Write Time"] + exec_time["Read Time"]

ram_time = exec_time._get_value(0, "Total")

exec_time = exec_time.drop(index=(0))

pf = exec_time[["Total"]] / ram_time

ax = pf.plot.bar(stacked=True, grid=True)

ax.axhline(y=1, color='r', linestyle='--', label="RAM Execution Time")

ax.set_ylabel('Time [s]')
ax.set_xlabel(x_label)
ax.set_title(title + ' - ' + operator + ' - ' + experiment)

labels = list(throughput.index.values)
ax.set_xticklabels(labels, rotation=0)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

png = png_folder + "/" + output + ".png"
tex = tex_folder + "/" + output + ".tex"

plt.savefig(png, dpi=350, bbox_inches='tight')
tikzplotlib.save(tex)