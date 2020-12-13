import numpy as np
import dateutil
import itertools
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import MDAnalysis
from MDAnalysis.analysis import align, rms, distances
sns.set(
    style="white",
    #context="paper",
    #font_scale=1.2,
    rc={
        "axes.facecolor": (0, 0, 0, 0), 
        "legend.facecolor": "white", 
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.labelpad": 6,
    }
)
        
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

LABEL_FONTSIZE=20
TICK_FONTSIZE=15
ALPHA1 = 0.55
BINS = 21
FIGSIZE1=(7.5, 5)
LEGEND_FACECOLOR = "white"

BLACK='k'
C1, C2, C3, C4, C5, C6 = sns.color_palette()[:6]

def save_fig(name):
    here = Path(__file__).resolve().parent
    filename = here.joinpath(f"figures/{name}.pdf")
    plt.savefig(filename, bbox_inches="tight", dpi=600)

def parse_datetime(line):
    if '|' in line:
        line = line.split('|')[0]
    return dateutil.parser.parse(line)

def get_experiment_start_time(experiment_dir):
    """Return the datetime object of when experiment began"""
    with open(Path(experiment_dir).joinpath("experiment_main.log")) as fp:
        line = fp.readline()
        assert "Logging on" in line, f"{line}"
    return parse_datetime(line)

def get_training_times(experiment_dir):
    """Return a list of [(train_start, train_end)] expressed in elapsed seconds."""
    p = Path(experiment_dir)
    if p.joinpath("training.log").is_file():
        return _parse_gpu_train_logs(p.joinpath("training.log"))
    elif p.joinpath("run.log").is_file():
        return _parse_cs1_train_logs(p.joinpath("run.log"))
    elif p.joinpath("cs1_run.log").is_file():
        return _parse_cs1_train_logs(p.joinpath("cs1_run.log"))
    else:
        raise ValueError("No training log file found")

def _parse_cs1_train_logs(p):
    # Training finished with 3276800 samples in 71.159 seconds
    durations = []
    print(f"Parsing CS1 training log: {p}")
    with open(p) as fp:
        for line in fp:
            if "Training finished with" in line:
                assert line.split()[-5] == "seconds,"
                t = float(line.split()[-6])
                durations.append(t)

    checkpoint_write_times = []
    t0 = get_experiment_start_time(p.parent)
    for chk in p.parent.joinpath("model_weights").glob("model.ckpt*.index"):
        chk_time = datetime.fromtimestamp(chk.stat().st_mtime)
        chk_time = (chk_time - t0).total_seconds()
        checkpoint_write_times.append(chk_time)

    training_time_intervals = [
        (chk_time - t, chk_time) for chk_time, t in zip(sorted(checkpoint_write_times), durations)
    ]
    return training_time_intervals

def _parse_gpu_train_logs(p):
    print(f"Parsing GPU training log: {p}")
    t0 = get_experiment_start_time(p.parent)
    starts = []
    ends = []
    with open(p) as fp:
        for line in fp:
            if "start train" in line:
                starts.append(parse_datetime(line))
            elif "end train" in line:
                ends.append(parse_datetime(line))

    training_time_intervals = [
        ((start-t0).total_seconds(), (end-t0).total_seconds())
        for start, end in zip(starts, ends)
    ]
    return training_time_intervals

def _get_lof_internal_timings(experiment_dir):
    # Outlier detection:
    # start model prediction with weights (begin fwd pass)
    # end model prediction (end of fwd pass)

    # Starting outlier searching (start of LOF)
    # Done with outlier searching (end of LOF)

    # Sending new outlier pdbs to (start of IO)
    # Finished sending new outliers (end of IO)
    fwd_pass_times = [[], []]
    lof_times = [[], []]
    io_times = [[], []]

    with open(Path(experiment_dir).joinpath('outlier_detection.log')) as fp:
        for line in fp:
            if "start model prediction with weights" in line:
                fwd_pass_times[0].append(parse_datetime(line))
            elif "end model prediction" in line:
                fwd_pass_times[1].append(parse_datetime(line))
            elif "Starting outlier searching" in line:
                lof_times[0].append(parse_datetime(line))
            elif "Done with outlier searching" in line:
                lof_times[1].append(parse_datetime(line))
            elif "Sending new outlier pdbs" in line:
                io_times[0].append(parse_datetime(line))
            elif "Finished sending new outliers" in line:
                io_times[1].append(parse_datetime(line))
    t0 = get_experiment_start_time(experiment_dir)
    for item in (fwd_pass_times, lof_times, io_times):
        for dim in (0,1):
            item[dim] = [(t-t0).total_seconds() for t in item[dim]]
    return fwd_pass_times, lof_times, io_times


def get_lof_times(experiment_dir):
    fwd_pass_times, lof_times, io_times = _get_lof_internal_timings(experiment_dir)
    outlier_detection_intervals = list(zip(fwd_pass_times[0], io_times[1]))
    return outlier_detection_intervals

def draw_lof_profile(experiment_dir, ax=None, legend_loc=2, legend_pos=None, title=None):
    fwd_pass_times, lof_times, io_times = _get_lof_internal_timings(experiment_dir)
    fwd_pass_durations = [fwd_pass_times[1][i] - fwd_pass_times[0][i] for i in range(len(fwd_pass_times[1]))][1:]
    lof_durations = [lof_times[1][i] - lof_times[0][i] for i in range(len(lof_times[1]))][1:]
    io_durations = [io_times[1][i] - io_times[0][i] for i in range(len(io_times[1]))][1:]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'CVAE', 'LOF', 'I/O'
    sizes = [sum(fwd_pass_durations), sum(lof_durations), sum(io_durations)]
    print(f"Total time in {labels} is {sizes}")
    total = sum(sizes)
    percents = [size / total for size in sizes]
    explode = (0, 0.0, 0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')

    if ax is None:
        fig1, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    patches, texts, *_ = ax.pie(sizes, explode=explode, shadow=True, startangle=90)
    labels = [f'{label} - {percent*100:.1f}%' for label, percent in zip(labels, percents)]
    ax.legend(patches, labels, facecolor=LEGEND_FACECOLOR, loc=legend_loc, bbox_to_anchor=legend_pos)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

def get_sim_running_count(exp_dir):
    sim_starts, sim_ends = [], []
    print("Scanning md_runs for start/end times (this takes a while...)")
    for fname in Path(exp_dir).joinpath("md_runs").glob("run0???.log"):
        with open(fname) as fp:
            for line in fp:
                if "START sim.step" in line:
                    sim_starts.append(parse_datetime(line))
                elif "END sim.step" in line:
                    sim_ends.append(parse_datetime(line))
                    
    t0 = get_experiment_start_time(exp_dir)
    sim_starts = [(t-t0).total_seconds() for t in sim_starts]
    sim_ends = [(t-t0).total_seconds() for t in sim_ends]
    combined = [(t, 1) for t in sim_starts] + [(t, -1) for t in sim_ends]
    combined = sorted(combined, key = lambda x: x[0])
    #times = [x[0] for x in combined]
    #counts = [x[1] for x in combined]
    times, counts = zip(*combined)
    counts = np.clip(np.cumsum(counts), 0, 160)
    return times, counts

def experiment_timeline(exp_dir, y_range=40, top_pad=5):
    train_intervals = get_training_times(exp_dir)
    #print("Training:", train_intervals)
    lof_intervals = get_lof_times(exp_dir)
    #print("LOF:", lof_intervals)
    md_times, md_counts = get_sim_running_count(exp_dir)

    fig = plt.figure(figsize=FIGSIZE1)
    ax = plt.subplot()

    exp_end = max(md_times)/60 + 5
    ax.step([t/60 for t in md_times], md_counts, where='post', label="MD Run Count")

    ax.set_xlabel("Experiment time (minutes)")
    ax.set_ylabel("Running MD Count")
    
    y_top = max(md_counts) + top_pad
    y_bottom = y_top - y_range
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(0, exp_end)
    #ax.set_yticks([130,140,150,160])
       
    # These spans are scaled by the width of the canvas:
    for i, (start, end) in enumerate(sorted(train_intervals)):
        ax.axhspan(y_bottom+12, y_bottom+18, xmin=start/60/exp_end, xmax=end/60/exp_end, color=C2, ec=BLACK)
    ax.plot(1, y_top+10, lw=4, color=C2, label="Train")
    for i, (start, end) in enumerate(sorted(lof_intervals)):
        ax.axhspan(y_bottom+2, y_bottom+8, xmin=start/60/exp_end, xmax=end/60/exp_end, color=C3, ec=BLACK)
    ax.plot(1, y_top+10, lw=4, color=C3, label="LOF")
    ax.set_ylim(y_bottom, y_top)
    ax.legend(bbox_to_anchor=(0.015, 0.99), loc=2, borderaxespad=0., ncol=3, columnspacing=1, labelspacing=-1)

def parse_loss(exp_dir):
    p = Path(exp_dir)
    if p.joinpath("training.log").is_file():
        fname = p.joinpath("training.log")
        return _parse_gpu_loss(fname)
    elif p.joinpath("run.log").is_file():
        fname = p.joinpath("run.log")
        return _parse_cs1_loss(fname)
    elif p.joinpath("cs1_run.log").is_file():
        fname = p.joinpath("cs1_run.log")
        return _parse_cs1_loss(fname)
    else:
        raise ValueError(f"No train log in {p}")
        
def _parse_gpu_loss(fname):
    steps, losses, times = [], [], []
    with open(fname) as f:
        for line in f:
            if "loss =" in line:
                tokens = line.split()
                time = parse_datetime(line)
                step = int(tokens[8]) # '32'
                loss = float(tokens[5][:-1]) # '18780.844,
                steps.append(step)
                losses.append(loss)
                times.append(time)
    t0 = times[0]
    assert t0 == min(times)
    times = [(t-t0).total_seconds() for t in times]
    deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
    big_deltas = [(i, d) for i, d in enumerate(deltas) if d > 30]
    for i,d in big_deltas:
        for j in range(i+1, len(times)):
            times[j] -= d
    return steps, losses, times

def _parse_cs1_loss(fname):
    steps, losses, times = [0], [float("inf")], [0]
    steps_per_sec = []
    idx = 1
    with open(fname) as f:
        for line in f:
            if "loss =" in line and "Saving dict" not in line:
                tokens = line.split()
                step = int(tokens[2][:-1]) # '1216:'
                loss = float(tokens[5]) # 4780.0
                per_sec = float(tokens[6][1:])
                time = (step - steps[idx-1]) / per_sec
                #print(f"{step - steps[idx-1]} took {time} sec")
                steps.append(step)
                losses.append(loss)
                times.append(time)
                idx += 1
    times = list(np.cumsum(times))
    return steps, losses, times

def plot_losses(exp_dirs, labels=None, colors=None, scale=1.0, as_time=False, log_scale=True):
    fig = plt.figure(figsize=FIGSIZE1)
    ax = plt.subplot()

    exps = [parse_loss(d) for d in exp_dirs]
    if colors is None:
        colors = [C1,C2,C3,C4,C5,C6]
    if log_scale:
        plt.yscale("log")
    if labels is None:
        labels = [d.name for d in exp_dirs]

    for label, (steps, losses, times), color in zip (labels, exps, colors):
        x_value = [t/60.0 for t in times] if as_time else steps
        losses = np.array(losses) * scale
        ax.plot(x_value, losses, label=label, c=color)
    ax.set_xlabel("Cumulative Training Time / min" if as_time else "Training steps")
    ax.set_ylabel("Normalized Training Loss")
    ax.legend(facecolor="white")

def plot_conformational_sampling(paths, iterations=[0], xlabel="", ylabel=""):

    # Collect all RMSD from each experiment
    rmsd_files = []
    for path in paths:
        rmsd_files.extend(list(path.joinpath("outlier_runs").glob("rmsds-*")))

    # Sort RMSD files by time of creation
    rmsd_files = sorted(
        rmsd_files, # rmsds-1607655171.npy
        key=lambda p: int(p.with_suffix("").name.split("-")[1])
    )
    
    # Load RMSD arrays into memory
    rmsd_values = [np.load(f.as_posix()) for f in rmsd_files]
    
    # Plot histograms
    for ind, color in zip(
        sorted(iterations),
        itertools.cycle(sns.color_palette()
    )):
        plt.hist(
            rmsd_values[ind],
            bins=BINS,
            label=f"iter-{ind}",
            color=color,
            alpha=ALPHA1,
        )

    plt.legend(facecolor=LEGEND_FACECOLOR)
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

def kde_conformational_sampling(paths, iterations=[0], labels=None, xlabel="", ylabel="", aspect=2.0):

    # Collect all RMSD from each experiment
    rmsd_files = []
    for path in paths:
        rmsd_files.extend(list(path.joinpath("outlier_runs").glob("rmsds-*")))

    # Sort RMSD files by time of creation
    rmsd_files = sorted(
        rmsd_files, # rmsds-1607655171.npy
        key=lambda p: int(p.with_suffix("").name.split("-")[1])
    )
    
    # Load RMSD arrays into memory
    rmsd_values = [np.load(f.as_posix()) for f in rmsd_files]
    
    # Plot KDEs
    iterations = sorted(iterations)
    if labels is None:
        labels = [f"iter-{ind}" for ind in iterations]
    data = {label: rmsd_values[ind] for (label, ind) in zip(labels, iterations)}
    ax = sns.displot(data, kind="kde", palette="icefire", legend=False, aspect=aspect, bw_adjust=0.1, gridsize=1000, clip=(2,10), fill=True, common_norm=False)
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    return ax

def kde_run_comparison(paths, iteration, labels=None, xlabel="", ylabel="", legend_pos=(0.7, 0.8), aspect=1.6):
    # Collect all RMSD from each experiment
    rmsd_file = {}
    rmsd_values = {}
    for path in paths:
        rmsd_files = list(path.joinpath("outlier_runs").glob("rmsds-*"))
        rmsd_file[path] = sorted(
            rmsd_files,
            key=lambda p: int(p.with_suffix("").name.split("-")[1])
        )[iteration]

    
    # Load RMSD arrays into memory
    if labels is None:
        labels = [p.name for p in paths]
    for path, label in zip(paths, labels):
        rmsd_values[label] = np.load(rmsd_file[path].as_posix())
        print(label, "is from", rmsd_file[path], "mean", np.mean(rmsd_values[label]))
        
    g = sns.displot(rmsd_values, kind="kde", palette="icefire", fill=True, legend=False, common_norm=False, aspect=aspect, bw_adjust=0.1, gridsize=1000)
    g.ax.set_xlim(3, 9)
    label_dict = dict(zip(labels, reversed(g.ax.collections)))
    g.add_legend(label_dict, loc='right', bbox_to_anchor=legend_pos)
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)

def fraction_of_contacts(cm, ref_cm):
    """
    Given two contact matices of equal dimensions, computes
    the fraction of entries which are equal. This is
    comonoly refered to as the fraction of contacts and
    in the case where ref_cm represents the native state
    this is the fraction of native contacts.
    Parameters
    ----------
    cm : np.ndarray
        A contact matrix.
    ref_cm : np.ndarray
        The reference contact matrix for comparison.
    """
    return 1 - (cm != ref_cm).mean()

def compute_contact_fractions(
    u: MDAnalysis.Universe,
    u_ref: MDAnalysis.Universe,
    cutoff: float = 8.0,
    select: str = "protein and name CA",
) -> np.ndarray:
    # Atom selection
    atoms = u.select_atoms(select)
    # Get atomic coordinates of reference atoms
    ref_positions = u_ref.select_atoms(select).positions.copy()

    # Get contact map of reference atoms
    ref_cm = distances.contact_matrix(
        ref_positions, cutoff, returntype='sparse'
    )

    contacts = []

    for _ in u.trajectory:
        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions
        # Compute contact map of current frame
        cm = distances.contact_matrix(positions, cutoff, returntype='sparse')
        # Compute fraction of contacts to reference state
        contacts.append(fraction_of_contacts(cm, ref_cm))

    return np.array(contacts)

def analyze_traj(
    pdb_file: str,
    dcd_file: str,
    reference_path: str,
    select: str = "protein and name CA",
    cutoff: float = 8.0,
    in_memory: bool = True,
):
    
    u = MDAnalysis.Universe(pdb_file, dcd_file, in_memory=in_memory)
    u_ref = MDAnalysis.Universe(
        reference_path, in_memory=in_memory
    )

    # Align trajectory to reference structure
    align.AlignTraj(
        u, u_ref, select=select, in_memory=in_memory,
    ).run()

    # Compute RMSD to reference
    rmsds = rms.RMSD(
        u, u_ref, select=select, in_memory=in_memory,
    ).run().rmsd

    # Compute fraction of contacts to reference
    contacts = compute_contact_fractions(
        u, u_ref, cutoff, select,
    )

    data = [{
        "rmsd": float(rmsd[2]),
        "contact": float(contact),
        "frame": int(rmsd[0]),
        "pdb_file": pdb_file,
        "dcd_file": dcd_file,
    } for rmsd, contact in zip(rmsds, contacts)]

    return data

def serialize_md_runs(experiment_dir: Path):
    
    # Get MD run dirs sorted Stream-AI-MD iteration
    md_run_dirs = sorted(
        filter(
            lambda p: p.is_dir() and "input" not in p.name,
            experiment_dir.joinpath("md_runs").iterdir(),
        ),
        key=lambda p: int(p.name.split("_")[1])
    )

    return md_run_dirs

def stream_ai_md_iterations(experiment_dir: Path):
    md_run_dirs = serialize_md_runs(experiment_dir)
    iterations = defaultdict(list)
    for md_run_dir in md_run_dirs:
        run_iter = md_run_dir.name.split("_")[1]
        iterations[run_iter].append(md_run_dir)
    # Get list of md_run_dirs for each iteration of Stream-AI-MD
    return [val for val in iterations.values()]

def _worker(kwargs):
    return analyze_traj(**kwargs)

def analyze_md_runs(
    experiment_dir: Path,
    reference_path: Path,
    select: str = "protein and name CA",
    cutoff: float = 8.0,
    in_memory: bool = True,
    num_workers: int = 10,
):
    # Collect all md run directories
    md_run_dirs = serialize_md_runs(experiment_dir)
        
    all_data = []
    
    kwargs = [{
        "pdb_file": next(md_run_dir.glob("*.pdb")).as_posix(),
        "dcd_file": next(md_run_dir.glob("*.dcd")).as_posix(),
        "reference_path": reference_path.as_posix(),
        "select": select,
        "cutoff": cutoff,
        "in_memory": in_memory,
    } for md_run_dir in md_run_dirs]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for data in tqdm(executor.map(_worker, kwargs)):
            all_data.append(data)

    return all_data

def dist_pbc(a, b, box):
    """
    calculate distance between two points in PBC box
    """
    assert len(a) == len(b)
    box = box[:len(a)]
    a = a % box
    b = b % box
    dist_vec = np.abs(a - b)
    dist_vec = np.abs(dist_vec - box * (dist_vec > box/2))
    return np.linalg.norm(dist_vec)

def _compute_com(
    pdb_file: str,
    dcd_file: str,
    select: str = "protein and name CA",
    in_memory: bool = True,
    sub_sample: int = 1,
):

    u = mda.Universe(
        pdb_file, dcd_file, in_memory=in_memory
    )

    protein_ca = u.select_atoms(select)
    nsp16 = u.segments[0].atoms
    nsp10 = u.segments[1].atoms

    dists = []
    for _ in u.trajectory[::sub_sample]: 
        dists.append(
            dist_pbc(
                nsp10.center_of_mass(), 
                nsp16.center_of_mass(), 
                nsp16.dimensions,
            )
        )

    sub_sampled_frames = np.arange(0, len(u), sub_sample)
    data = [{
        "dist": dist,
        "frame": frame,
        "pdb_file": pdb_file,
        "dcd_file": dcd_file,
    } for dist, frame in zip(dists, sub_sampled_frames)]
    
    return data

def _com_worker(kwargs):
    return _compute_com(**kwargs)

def compute_center_of_mass_dist(
    experiment_dir: Path,
    select: str = "protein and name CA",
    in_memory: bool = True,
    sub_sample: int = 1,
    num_workers: int = 10,
):
    # Collect all md run directories
    md_run_dirs = serialize_md_runs(experiment_dir)
    
    all_data = []
    
    kwargs = [{
        "pdb_file": next(md_run_dir.glob("*.pdb")).as_posix(),
        "dcd_file": next(md_run_dir.glob("*.dcd")).as_posix(),
        "select": select,
        "sub_sample": sub_sample,
        "in_memory": in_memory,
    } for md_run_dir in md_run_dirs]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for data in tqdm(executor.map(_com_worker, kwargs)):
            all_data.append(data)
    
    return all_data

def write_pdb(output_pdb, input_pdb, traj_file, frame):
    mda_traj = MDAnalysis.Universe(input_pdb, traj_file)
    mda_traj.trajectory[frame]
    PDB = MDAnalysis.Writer(output_pdb)
    PDB.write(mda_traj.atoms)
