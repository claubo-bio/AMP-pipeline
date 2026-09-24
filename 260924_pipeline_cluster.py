""" 
## Setup

### CM15 Scaffold Design Pipeline

**Starting point:** 19 NMR structures

**Pipeline**

`19 NMR structures`
→
`Rosetta Remodel`
→
`RFD3 partial diffusion`
→
`RFD3 scaffold generation`
→
`LigandMPNN sequence design`
→
`Boltz structure prediction`
→
`Rosetta scoring / relaxation`
→
`LigandMPNN → Boltz → Rosetta`
→
`...`

### Design strategy

* Focus on the **N-terminal linker/cleavage region**, also because it looks structurally more appropriate; the C-terminal linker is currently lower priority because it would leave a longer overhang.
* Introduce the cleavage region using **Rosetta Remodel**, with **loop or helix** preferences depending on the input. (first **loop**)
* Use **RFD3 partial diffusion** to introduce structural variation while retaining the desired CM15 geometry. (with 5A the helix stays intact, but could also test 10A)
* Generate variable-length scaffolds with **RFD3**.
* Use **LigandMPNN** to design the scaffold sequence while keeping CM15 fixed.
* Predict designed structures with **Boltz**.
* Evaluate structural quality with **Rosetta**.
* Repeat the **MPNN → Boltz → Rosetta** cycle for iterative refinement.(2-3x?)

### Metrics

**RFD3**

mostly documentation:
* Number of residues
* Chainbreaks (filter = 0)
* Backbone / side-chain clashes (filter = 0)
* Radius of gyration ((filter = depends on length))
* Secondary-structure composition
* CA-deviation (this is the change from initial coordinates?)

**LigandMPNN / Boltz**

* MPNN confidence
* Boltz confidence (filter or rank?)
* pLDDT
* Other structure-confidence metrics

**Rosetta**

Filter:
actual values still unclear
* Total SAP score, SAP per Res, SAP helix
* Cleavage-site SASA
* Total energy score (just documentation)
* Energy Score per residue
* BUNS (Sc buns)
* PackStat

"""

# ============================================================ # Imports # ============================================================
import protflow
from protflow.poses import Poses

# Jobstarters determine HOW ProtFlow executes computational jobs. 
# # LocalJobStarter runs processes on the current machine. 
# # SbatchArrayJobstarter can submit jobs to a SLURM cluster.
from protflow import jobstarters
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.residues import residue_selection # Used later for selecting specific residues, e.g. A1-A15.

from protflow.tools.rfdiffusion3 import RFdiffusion3, RFD3Params


import logging
import os
import time
import pandas as pd

# ============================================================ # Logging # ============================================================
# Configure the format of messages printed by ProtFlow and the Python logging system.
# This is useful for monitoring longer computational runs: 
# instead of only seeing that something is running, we can see # when jobs start, finish, and whether warnings/errors occur.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# ============================================================ # Jobstarter # ============================================================
# defining jobstarter, in this case a local jobstarter that will run jobs in parallel on the local machine
# local_jobstarter = LocalJobStarter(max_cores=2)

#SbatchArrayJobstarter runs jobs by submitting a SLURM job
gpu_jobstarter = SbatchArrayJobstarter(max_cores=10, gpus=1, options="--time=24:10:00 ")
rosetta_jobstarter = SbatchArrayJobstarter(
    max_cores=4,
    gpus=0,
    options="--time=24:10:00"
)

# ============================================================ # Output directory # ============================================================
# Using a dedicated output directory keeps the input structures,  RFdiffusion results, and later sequence-design results  organized separately.
OUT_ROOT  = 'cm15_data/outputs/'
os.makedirs(OUT_ROOT, exist_ok=True) ## Create the directory if it does not already exist.

#some messages how the output directory is set up
print('Setup complete.')
print(f'  Output root: {os.path.abspath(OUT_ROOT)}') # converts the relative path above into the # full path on the current computer
# ============================================================ # Alternative Input from previous runs # ============================================================
###extra step if you want to reload the dataframe of the output poses from the CSV file, instead of using the dataframe in memory:

""" 

scored_poses2 = Poses(
    poses="/home/aektakaerlek/CM15_project/AMP-pipeline/cm15_data/outputs/260918_cm15_TEV_Remodel_Nterminal_loop_partial_RFD_MPNN_Boltz_rosfilters/260918_cm15_TEV_Remodel_Nterminal_loop_partial_RFD_MPNN_Boltz_rosfilters_scores.csv",
    work_dir="/home/aektakaerlek/CM15_project/AMP-pipeline/cm15_data/outputs/260918_cm15_TEV_Remodel_Nterminal_loop_partial_RFD_MPNN_Boltz_rosfilters",
    storage_format="csv",
    jobstarter=gpu_jobstarter,
)

print(f"Loaded {len(scored_poses2.df)} poses")
"""

# Defining Input and Poses

# ============================================================ # Create the initial Poses object # ============================================================
# # Define a dedicated output directory for this run. 
OUT_DENOVO = os.path.join(OUT_ROOT, '260924_cm15_TEV_Remodel_Nterminal_loop_partial_RFD_MPNN_Boltz_rosfilters') #change the name of the output directory to something meaningful for your experiment
os.makedirs(OUT_DENOVO, exist_ok=True) # Create the directory if it does not already exist.


# Create a ProtFlow Poses object from the input PDB structures. 
# # # Here, 'poses' points to a DIRECTORY containing PDB files. 
# # ProtFlow searches this directory using the glob pattern below.
## The CSV will store also store scores and other data also from later steps like LigandMpnn and Structure Prediction. 
poses_input = Poses(
    poses='cm15_data/inputs_TEV_loop/',
    glob_suffix='*pdb',
    work_dir=OUT_DENOVO,
    storage_format='csv',
    jobstarter=gpu_jobstarter,
)

# ============================================================ # Save the motif # ============================================================
from protflow.residues import residue_selection

poses_input.df["cm15_motif"] = [
    residue_selection(
        ",".join(f"A{i}" for i in range(1, 23)),
        delim=","
    )
    for _ in poses_input.poses_list()
]

# Inspect the dataframe underlying the Poses object.
print(poses_input.df)
print(f'Poses created: {len(poses_input.df.index)} rows (expected x)') ## Check that the number of rows matches the number of input PDB files.

# Setting Up RFDiffusion3

# ============================================================ # Configure and run RFdiffusion3 with Partial-Diff # ============================================================
# The runner is responsible for launching RFdiffusion3 and handling the input/output files.
# We can reuse the same runner for multiple RFdiffusion3 calculations.
rfdiffusion3 = RFdiffusion3(jobstarter=gpu_jobstarter)

# RFD3Params stores the design instructions that RFdiffusion3 should apply to each input pose.
params = RFD3Params(poses=poses_input) #change to e.g. relaxed_poses if you want to use the relaxed poses instead of the original input poses
params.set_input_specs(
    contig="A1-22",
    partial_t=5.0, # add 5A noise to the motif region 
    select_fixed_atoms=False)

# ------------------------------------------------------------
# Run Partial Diffusion
# ------------------------------------------------------------

poses_partial = rfdiffusion3.run(
    poses=poses_input, #change to relaxed_poses if you want to use the relaxed poses instead of the original input poses
    prefix='partial_diff',
    params=params,
    n_batches=2, # How many batches of diffusion to perform 
    diffusion_batch_size=2, # How many designs are generated in each batch
    overwrite=True, #Allows existing output files from an earlier run to be # overwritten.
)

print(f'Partial diffusion run complete: {len(poses_partial.df.index)} output poses')

#pretty quick, only a few seconds per pose. for 38 poses the run took 5min

# =========================================================== # Configure and run RFdiffusion3 # ============================================================

# RFD3Params stores the design instructions that RFdiffusion3 should apply to each input pose.
params = RFD3Params(poses=poses_partial) #change to e.g. relaxed_poses if you want to use the relaxed poses instead of the original input poses
params.set_input_specs(
    contig="40-120,A1-22", 
    select_fixed_atoms="A8-22",
    select_exposed="A1-7",
    select_buried="A8-22")

# ------------------------------------------------------------ # Run RFdiffusion3 # ------------------------------------------------------------
# Example if there are 19 input PDB poses and 2 diffusion samples per pose:
# 38 input poses × 2 designs per pose = 76 output poses. (x2 batches = 152)
poses_denovo = rfdiffusion3.run(
    poses=poses_partial, #change to relaxed_poses if you want to use the relaxed poses instead of the original input poses
    prefix='rfdiffusion3',
    params=params,
    n_batches=3, # How many batches of diffusion to perform -> per batch a certain length is defined
    diffusion_batch_size=2,# How many designs are generated in each batch
    update_motifs=["cm15_motif"], # implemented motif save
    overwrite=True, #Allows existing output files from an earlier run to be # overwritten.
)

print(f'De novo run complete: {len(poses_denovo.df.index)} output poses')

print(poses_denovo.df.columns.tolist()) #see all the columns of the dataframe of the output poses

##previous test took 45min for 38 structures on a local machine with 2 cores.
# or 150min for 152 poses, another test 126 min for 152.#300 min for 300 poses
  
#16 min for 200 poses on acluster, 25min 300 poses

## Inspection of RFD3 run results.
# ------------------------------------------------------------ # Plotting of RFdiffusion3 results # ------------------------------------------------------------
from protflow.utils.plotting import violinplot_multiple_cols

#print(poses_denovo.df) #print the dataframe of the output poses
#print(poses_denovo.df.columns.tolist()) #see all the columns of the dataframe of the output poses

## see some useful metrics of the output poses, 
for _, pose in poses_denovo.df.iterrows():
    print(
        pose[
            [
                "poses_description",
                "rfdiffusion3_num_residues",
                "rfdiffusion3_helix_fraction",
                "rfdiffusion3_sheet_fraction",
                "rfdiffusion3_loop_fraction",
                "rfdiffusion3_n_chainbreaks",
                "rfdiffusion3_max_ca_deviation", #says something about ca deviation of the fixed motif?
                "rfdiffusion3_radius_of_gyration",
                'rfdiffusion3_n_clashing.interresidue_clashes_w_sidechain',
                'rfdiffusion3_n_clashing.interresidue_clashes_w_backbone',        
            ]
        ]
    )
    

violinplot_multiple_cols(
    dataframe=poses_denovo.df,
    cols=[
        "rfdiffusion3_num_residues",
        "rfdiffusion3_max_ca_deviation",
        "rfdiffusion3_radius_of_gyration",
        "rfdiffusion3_helix_fraction",
        "rfdiffusion3_sheet_fraction",
        "rfdiffusion3_loop_fraction",
        "rfdiffusion3_n_chainbreaks",
        'rfdiffusion3_n_clashing.interresidue_clashes_w_sidechain',
        'rfdiffusion3_n_clashing.interresidue_clashes_w_backbone',        
    ],

    y_labels=[
        "Number of residues",
        "RFD3 max CA deviation (Å)",
        "Radius of gyration (Å)",
        "Helix fraction",
        "Sheet fraction",
        "Loop fraction",
        "Number of chainbreaks",
        "Sidechain clashes",
        "Backbone clashes",
    ],
    titles=[
        "RFD3 length",
        "RFD3 fixed motif",
        "RFD3 compactness",
        "RFD3 secondary str",
        "RFD3 secondary str",
        "RFD3 secondary str",
        "RFD3 chain continuity",
        "RFD3 clashes sc",
        "RFD3 clashes bb",
    ],
    out_path=os.path.join(
        poses_denovo.plots_dir,
        "rfd3_boltz_metrics.png"
    )
)
## Filtering
# ------------------------------------------------------------ # Filtering # ------------------------------------------------------------
#Filter out pose with chain breaks and prefer more compact structures. 
#Or anything else you are interested in.


poses_denovo.filter_poses_by_value(
    score_col="rfdiffusion3_n_chainbreaks",
    value=0,
    operator="=",
    prefix="rfd3_chainbreaks",
    plot=True
)

poses_denovo.filter_poses_by_value(
    score_col="rfdiffusion3_radius_of_gyration",
    value=16,
    operator="<=",
    prefix="rfd3_gyration",
    plot=True
)

poses_denovo.filter_poses_by_value(
    score_col="rfdiffusion3_n_clashing.interresidue_clashes_w_sidechain",
    value=0,
    operator="=",
    prefix="rfd3_clashes_sc",
    plot=True
)

# Sequence Design for diffused poses with LigandMPNN
"""
poses_denovo.df["fixed_residues"] = (
    "A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15"
)
""" 

# ------------------------------------------------------------ # Setting up LigandMPNN # ------------------------------------------------------------
from protflow.tools import ligandmpnn
from protflow.tools.ligandmpnn import LigandMPNN
#import inspect

# setup mover
ligandmpnn_runner = LigandMPNN(jobstarter=gpu_jobstarter)

# inspect run arguments
# print(inspect.signature(ligandmpnn_runner.run))


# ################# sanity check for motif fix ####################

print(
    poses_denovo.df[
        [
            "poses_description",
            "cm15_motif",
        ]
    ].head()
)

# ------------------------------------------------------------ # Start LigandMPNN # ------------------------------------------------------------
# design xx sequences per Pose with LigandMPNN
mpnn_designs = ligandmpnn_runner.run( 
    poses=poses_denovo,
    prefix="mpnn_design",
    jobstarter=gpu_jobstarter,
    nseq=4, #better to keep this low for testing, can be increased later, because this will be predicted with Boltz 
    model_type="soluble_mpnn", #unsure if ligand_mpnn or sol_mpnn
    fixed_res_col="cm15_motif", #this is the column in the dataframe of the Poses object that contains the fixed residues, which are the residues of the motif that should not be changed by LigandMPNN
    options="--seed 111"
)
##previous run took about 1min for 38 structures on a local machine with 2 cores.

# ------------------------------------------------------------ # Plotting # ------------------------------------------------------------
from protflow.utils.plotting import violinplot_multiple_cols

violinplot_multiple_cols(
    dataframe=mpnn_designs.df,
    cols=[
        "mpnn_design_overall_confidence", ## does this include ligand confidence, idk why ligand confidence is calculated here 
        "mpnn_design_seq_rec" ##not really useful here
    ],
    y_labels=[
        "SolMPNN overall confidence",
        "SolMPNN sequence recovery"
    ],
    titles=[
        "SolMPNN confidence",
        "SolMPNN sequence recovery"
    ],
    out_path=os.path.join(
        mpnn_designs.plots_dir,
        "mpnn_metrics.png"
    )
)
# Predict Designed Sequences using Boltz

from protflow.tools import boltz
from protflow.tools.boltz import Boltz

# ------------------------------------------------------------ # Boltz Prediction # ------------------------------------------------------------
# set up runner
#why is there a batch input for boltz
boltz_runner = Boltz(jobstarter=gpu_jobstarter)

# 19 input poses × 4 RFD3 designs × 4 MPNN sequences = 250 Boltz predictions. (after filtering)
# start predicting
predicted_proteins = boltz_runner.run(
    poses=mpnn_designs,
    prefix="boltz",
    msa_setting="empty",
    options="--output_format pdb", # dont use this if you have non protein ligands
    overwrite=True
)
#Adding the --use_potentials flag, Boltz uses an inference time potential that significantly improve the physical quality of the poses.
#options="--use_msa_server", # you may remove this so this calculation will be significantly faster, but accuracy tradeoff

#without MSA:
#for 1200 poses ...228 min, so still not really a difference, but at least didnt get an error with too many requests to the MSA server.
# for around 40 min, 904 poses on a cluster, 99min for 2612 poses

# ------------------------------------------------------------ # Plotting # ------------------------------------------------------------

# you may plot some important scores. 

from protflow.utils.plotting import violinplot_multiple_cols
violinplot_multiple_cols(
    dataframe=predicted_proteins.df,
    cols=[
        "boltz_confidence_score",
        "boltz_ptm",
        "boltz_complex_plddt",
        "boltz_complex_pde",
    ],
    y_labels=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz complex pLDDT",
        "Boltz complex PDE (Å)",
    ],
    titles=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz pLDDT",
        "Boltz PDE",
    ],
    out_path=os.path.join(
        predicted_proteins.plots_dir,
        "boltz_metrics.png"
    )
)
### Boltz confidence metrics

### pLDDT → local confidence; pTM → global fold confidence; PDE → predicted structural error; confidence score → overall confidence.


# ------------------------------------------------------------ # Filtering # ------------------------------------------------------------
#overall confidence score to filter the predicted structures.
#consider filters for pTM, pLDDT, and PDE as well, depending on the desired quality of the predicted structures?
#not completely sure what are good cutoffs or if I should just rank them?

predicted_proteins.filter_poses_by_value(
    score_col="boltz_confidence_score",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_confidence",
    plot=True
)

predicted_proteins.filter_poses_by_value(
    score_col="boltz_ptm",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_ptm",
    plot=True
)

predicted_proteins.filter_poses_by_value(
    score_col="boltz_complex_plddt",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_plddt",
    plot=True
)


# More Scoring & Filtering

# check the rmsd values of the predicted proteins and filter them at RMSD < 1.0 Å, 
# and plot the distribution of RMSD values before and after filtering.

from protflow.metrics.rmsd import BackboneRMSD

rmsd_calculator = BackboneRMSD(jobstarter=gpu_jobstarter)

rmsd_calculator.run(
    poses=predicted_proteins,
    prefix="rfd3_vs_boltz_rmsd",
    ref_col="rfdiffusion3_location"
)

print(
    predicted_proteins.df[
        ["poses_description", "rfd3_vs_boltz_rmsd_rmsd"]
    ]
)
from protflow.utils.plotting import violinplot_multiple_cols

predicted_proteins.filter_poses_by_value(
    score_col="rfd3_vs_boltz_rmsd_rmsd",
    value=1.0,
    operator="<",
    prefix="rfd3_vs_boltz_rmsd_filter",
    plot=True
)

from protflow.residues import ResidueSelection

# ============================================================
# Label the residues of the fixed part
# ============================================================

predicted_proteins.df["cleavage_selection"] = predicted_proteins.df[
    "cm15_motif"
].apply(
    lambda motif: ResidueSelection(
        motif.to_string(delim=" ").split()[:7]
    )
)

predicted_proteins.df["core_selection"] = predicted_proteins.df[
    "cm15_motif"
].apply(
    lambda motif: ResidueSelection(
        motif.to_string(delim=" ").split()[7:22]
    )
)

###check if it worked
print(predicted_proteins.df["cleavage_selection"].iloc[0])
print(
    predicted_proteins.df["cleavage_selection"]
    .iloc[0]
    .to_string(ordering="rosetta")
)

print(
    predicted_proteins.df["core_selection"]
    .iloc[0]
    .to_string(ordering="rosetta")
)

####create per pose rosetta options for the core and cleavage residues, that can be used in the rosetta design step later.

predicted_proteins.df["rosetta_options"] = predicted_proteins.df.apply(
    lambda row: (
        f"-parser:script_vars "
        f"core_residues={row['core_selection'].to_string(ordering='rosetta')} "
        f"cleavage_residues={row['cleavage_selection'].to_string(ordering='rosetta')}"
    ),
    axis=1
)

print(predicted_proteins.df["rosetta_options"].iloc[0])

from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

sap_protocol = "cm15_data/runners_auxiliary_scripts/sasa_sap_total.xml"

s_options = f"-parser:protocol {sap_protocol} -beta"

scored_poses1 = rosetta.run(
    poses=predicted_proteins,
    prefix="score_10",
    nstruct=1,
    options=s_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses1.df.columns.tolist())
print(scored_poses1.df.head())
from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

bun_protocol = "cm15_data/runners_auxiliary_scripts/bun_packstat.xml"

b_options = f"-parser:protocol {bun_protocol} -beta"

scored_poses = rosetta.run(
    poses=scored_poses1,
    prefix="score_11",
    nstruct=1,
    options=b_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses.df.columns.tolist())
print(scored_poses.df.head())

##runs equally slow on cluster than on local computer
from protflow.utils.plotting import violinplot_multiple_cols

import os

scored_poses.df["score_per_residue"] = (
    scored_poses.df["score_10_total_score"]
    / scored_poses.df["rfdiffusion3_num_residues"]
)

scored_poses.df["sasa_per_residue"] = (
    scored_poses.df["score_10_total_sasa_sasa"]
    / scored_poses.df["rfdiffusion3_num_residues"]
)

scored_poses.df["sap_per_residue"] = (
    scored_poses.df["score_10_total_sap_sap_score"]
    / scored_poses.df["rfdiffusion3_num_residues"]
)

violinplot_multiple_cols(
    dataframe=scored_poses.df,
    cols=[
        "score_10_sap_score",
        "score_10_cleavage_sasa_sasa",
        "score_10_total_sasa_sasa",
        "score_10_total_score",
        "score_per_residue",
        "sap_per_residue",
        "score_11_buns_foo",
        "score_11_packstat_foo",
    ],
    y_labels=[
        "SAP score",
        "Cleavage-site SASA",
        "Total SASA",
        "Total energy",
        "Avg Energy per residue",
        "SAP per residue",
        "Buried unsatisfied sc H-bonds",
        "PackStat",
    ],
    titles=[
        "Helix SAP",
        "Cleavage-site SASA",
        "Total SASA",
        "Total energy",
        "Energy per residue",
        "SAP per residue",
        "Buried unsatisfied H-bonds",
        "PackStat",
    ],
    out_path=os.path.join(
        scored_poses.plots_dir,
        "scores_sasa_sap_energy_buns_packstat.png"
    )
)

## or filter by a certain cutoff, e.g. SAP score < 1.0, and plot the distribution of SAP scores before and after filtering.

from protflow.utils.plotting import violinplot_multiple_cols

scored_poses.filter_poses_by_value(
    score_col="score_per_residue",
    value=-0.5,
    operator="<=",
    prefix="score_per_res_filter",
    plot=True
)

scored_poses.filter_poses_by_value(
    score_col="score_11_buns_foo",
    value=1.0,
    operator="<=",
    prefix="score_buns_filter",
    plot=True
)

scored_poses.filter_poses_by_value(
    score_col="score_11_packstat_foo",
    value=0.5,
    operator=">=",
    prefix="packstat_filter",
    plot=True
)


#Setting up Rossetta Fast Relax
from protflow.tools.rosetta import Rosetta
rosetta = Rosetta(jobstarter=rosetta_jobstarter, fail_on_missing_output_poses=True)

# relax poses
relax_protocol = "cm15_data/runners_auxiliary_scripts/fastrelax.xml" ##Need to be changed to relative path later. 
fr_options = f"-parser:protocol {relax_protocol} -beta" # define options for rosetta relax runs (beta weights, and path to relax xml)
relaxed_poses = rosetta.run(
    poses=scored_poses,
    prefix="fast_rlx_afterboltz",
    nstruct=1,
    options=fr_options,
    rosetta_application="rosetta_scripts.default.linuxgccrelease",
)  #default is 5 relax trajectories per pose

# ============================================================ # Configure and run RFdiffusion3 with Partial-Diff # ============================================================
# The runner is responsible for launching RFdiffusion3 and handling the input/output files.
# We can reuse the same runner for multiple RFdiffusion3 calculations.
rfdiffusion3 = RFdiffusion3(jobstarter=gpu_jobstarter)

# RFD3Params stores the design instructions that RFdiffusion3 should apply to each input pose.
params = RFD3Params(poses=relaxed_poses) #change to e.g. relaxed_poses if you want to use the relaxed poses instead of the original input poses
params.set_input_specs(
    partial_t=5.0, # helix stays rather intact like this, or more likely 
    select_fixed_atoms=False)

# ------------------------------------------------------------
# Run Partial Diffusion
# ------------------------------------------------------------

poses_partial2 = rfdiffusion3.run(
    poses=relaxed_poses, #change to relaxed_poses if you want to use the relaxed poses instead of the original input poses
    prefix='partial_diff_2',
    params=params,
    n_batches=2, # How many batches of diffusion to perform 
    diffusion_batch_size=2, # How many designs are generated in each batch
    overwrite=True, #Allows existing output files from an earlier run to be # overwritten.
)

print(f'Partial diffusion run complete: {len(poses_partial2.df.index)} output poses')

#pretty quick, only a few seconds per pose. for 38 poses the run took 5min
# ------------------------------------------------------------ # Setup fixed residues # ------------------------------------------------------------
#fixed residues were already defined above and we can reuse them here for the relaxed poses.

# ------------------------------------------------------------ # Start LigandMPNN # ------------------------------------------------------------
# design xx sequences per Pose with LigandMPNN
mpnn_designs2 = ligandmpnn_runner.run( 
    poses=poses_partial2,
    prefix="mpnn_design_2",
    jobstarter=gpu_jobstarter,
    nseq=5, #better to keep this low for testing, can be increased later, because this will be predicted with Boltz 
    model_type="soluble_mpnn", #unsure if ligand_mpnn or sol_mpnn
    fixed_res_col="cm15_motif",
    options="--seed 111"
)
# ------------------------------------------------------------ # Boltz Prediction # ------------------------------------------------------------
# set up runner
boltz_runner = Boltz(jobstarter=gpu_jobstarter)

# 19 input poses × 4 RFD3 designs × 4 MPNN sequences = 250 Boltz predictions. (after filtering)
# start predicting
predicted_proteins2 = boltz_runner.run(
    poses=mpnn_designs2,
    prefix="boltz_2",
    msa_setting="empty", 
    options="--output_format pdb",
    overwrite=True
)

# ------------------------------------------------------------ # Plotting # ------------------------------------------------------------

# you may plot some important scores. 

from protflow.utils.plotting import violinplot_multiple_cols
violinplot_multiple_cols(
    dataframe=predicted_proteins2.df,
    cols=[
        "boltz_2_confidence_score",
        "boltz_2_ptm",
        "boltz_2_complex_plddt",
        "boltz_2_complex_pde",
    ],
    y_labels=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz complex pLDDT",
        "Boltz complex PDE (Å)",
    ],
    titles=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz pLDDT",
        "Boltz PDE",
    ],
    out_path=os.path.join(
        predicted_proteins2.plots_dir,
        "boltz_metrics2.png"
    )
)
# ------------------------------------------------------------ # Filtering # ------------------------------------------------------------
#overall confidence score to filter the predicted structures.
#consider filters for pTM, pLDDT, and PDE as well, depending on the desired quality of the predicted structures?
#not completely sure what are good cutoffs or if I should just rank them?

predicted_proteins2.filter_poses_by_value(
    score_col="boltz_2_confidence_score",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_2_confidence",
    plot=True
)

predicted_proteins2.filter_poses_by_value(
    score_col="boltz_2_ptm",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_2_ptm",
    plot=True
)

predicted_proteins2.filter_poses_by_value(
    score_col="boltz_2_complex_plddt",
    value=0.8,  # example cutoff 
    operator=">=",
    prefix="boltz_2_plddt",
    plot=True
)


from protflow.metrics.rmsd import BackboneRMSD

rmsd_calculator = BackboneRMSD(jobstarter=gpu_jobstarter)

rmsd_calculator.run(
    poses=predicted_proteins2,
    prefix="boltz_vs_boltz2_rmsd",
    ref_col="boltz_location" 
)
## not sure if it makes sense these if i run e.g. the relax and partial-t right before, rmsd could just be very different
## and isnt this to a certain degree what we want?

print(
    predicted_proteins2.df[
        ["poses_description", "boltz_vs_boltz2_rmsd_rmsd"]
    ]
)
''' 
from protflow.utils.plotting import violinplot_multiple_cols

predicted_proteins2.filter_poses_by_value(
    score_col="boltz_vs_boltz2_rmsd_rmsd",
    value=1.0,
    operator="<",
    prefix="boltz_vs_boltz2_rmsd_filter2",
    plot=True
)
'''

from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

sap_protocol = "cm15_data/runners_auxiliary_scripts/sasa_sap_total.xml"

s_options = f"-parser:protocol {sap_protocol} -beta"

scored_poses20 = rosetta.run(
    poses=predicted_proteins2,
    prefix="score_110",
    nstruct=1,
    options=s_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses20.df.columns.tolist())
print(scored_poses20.df.head())

##################

from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

bun_protocol = "cm15_data/runners_auxiliary_scripts/bun_packstat.xml"

b_options = f"-parser:protocol {bun_protocol} -beta"

scored_poses2 = rosetta.run(
    poses=scored_poses20,
    prefix="score_111",
    nstruct=1,
    options=b_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses2.df.columns.tolist())
print(scored_poses2.df.head())

from protflow.utils.plotting import violinplot_multiple_cols

import os

scored_poses2.df["score_per_residue2"] = (
    scored_poses2.df["score_110_total_score"]
    / scored_poses2.df["rfdiffusion3_num_residues"]
)

scored_poses2.df["sasa_per_residue2"] = (
    scored_poses2.df["score_110_total_sasa_sasa"]
    / scored_poses2.df["rfdiffusion3_num_residues"]
)

scored_poses2.df["sap_per_residue2"] = (
    scored_poses2.df["score_110_total_sap_sap_score"]
    / scored_poses2.df["rfdiffusion3_num_residues"]
)

violinplot_multiple_cols(
    dataframe=scored_poses2.df,
    cols=[
        "score_110_sap_score",
        "score_110_cleavage_sasa_sasa",
        "score_110_total_sasa_sasa",
        "score_110_total_score",
        "score_per_residue2",
        "sap_per_residue2",
        "score_111_buns_foo",
        "score_111_packstat_foo",
    ],
    y_labels=[
        "SAP score",
        "Cleavage-site SASA",
        "Total SASA",
        "Total energy",
        "Avg Energy per residue",
        "SAP per residue",
        "Buried unsatisfied sc H-bonds",
        "PackStat",
    ],
    titles=[
        "Helix SAP",
        "Cleavage-site SASA",
        "Total SASA",
        "Total energy",
        "Energy per residue",
        "SAP per residue",
        "Buried unsatisfied H-bonds",
        "PackStat",
    ],
    out_path=os.path.join(
        scored_poses2.plots_dir,
        "scores_sasa_sap_energy_buns_packstat2.png"
    )
)

## or filter by a certain cutoff, e.g. SAP score < 1.0, and plot the distribution of SAP scores before and after filtering.

from protflow.utils.plotting import violinplot_multiple_cols

scored_poses2.filter_poses_by_value(
    score_col="score_per_residue2",
    value=-1.0,
    operator="<=",
    prefix="score_per_res_filter2",
    plot=True
)

scored_poses2.filter_poses_by_value(
    score_col="score_111_buns_foo",
    value=0.0,
    operator="<=",
    prefix="score_buns_filter2",
    plot=True
)

scored_poses2.filter_poses_by_value(
    score_col="score_111_packstat_foo",
    value=0.6,
    operator=">=",
    prefix="packstat_filter2",
    plot=True
)



#Setting up Rossetta Fast Relax
from protflow.tools.rosetta import Rosetta
rosetta = Rosetta(jobstarter=rosetta_jobstarter, fail_on_missing_output_poses=True)

# relax poses
relax_protocol = "cm15_data/runners_auxiliary_scripts/fastrelax.xml" ##Need to be changed to relative path later. 
fr_options = f"-parser:protocol {relax_protocol} -beta" # define options for rosetta relax runs (beta weights, and path to relax xml)
relaxed_poses2 = rosetta.run(
    poses=scored_poses2,
    prefix="fast_rlx_afterboltz_3_",
    nstruct=1,
    options=fr_options,
    rosetta_application="rosetta_scripts.default.linuxgccrelease",
)  
#default is 5 relax trajectories per pose

# ============================================================ # Configure and run RFdiffusion3 with Partial-Diff # ============================================================
# The runner is responsible for launching RFdiffusion3 and handling the input/output files.
# We can reuse the same runner for multiple RFdiffusion3 calculations.
rfdiffusion3 = RFdiffusion3(jobstarter=gpu_jobstarter)

# RFD3Params stores the design instructions that RFdiffusion3 should apply to each input pose.
params = RFD3Params(poses=relaxed_poses2) #change to e.g. relaxed_poses if you want to use the relaxed poses instead of the original input poses
params.set_input_specs(
    partial_t=5.0, # helix stays rather intact like this, or more likely 
    select_fixed_atoms=False)

# ------------------------------------------------------------
# Run Partial Diffusion
# ------------------------------------------------------------

poses_partial3 = rfdiffusion3.run(
    poses=relaxed_poses2, #change to relaxed_poses if you want to use the relaxed poses instead of the original input poses
    prefix='partial_diff_3',
    params=params,
    n_batches=2, # How many batches of diffusion to perform 
    diffusion_batch_size=2, # How many designs are generated in each batch
    overwrite=True, #Allows existing output files from an earlier run to be # overwritten.
)

print(f'Partial diffusion run complete: {len(poses_partial3.df.index)} output poses')


# ------------------------------------------------------------ # Start LigandMPNN # ------------------------------------------------------------
# design xx sequences per Pose with LigandMPNN

mpnn_designs3 = ligandmpnn_runner.run( 
    poses=poses_partial3,
    prefix="mpnn_design_3",
    jobstarter=gpu_jobstarter,
    nseq=5, #better to keep this low for testing, can be increased later, because this will be predicted with Boltz 
    model_type="soluble_mpnn", #unsure if ligand_mpnn or sol_mpnn
    fixed_res_col="cm15_motif",
    options="--seed 111"
)

# ------------------------------------------------------------ # Boltz Prediction # ------------------------------------------------------------
# set up runner
boltz_runner = Boltz(jobstarter=gpu_jobstarter)

# start predicting
predicted_proteins3 = boltz_runner.run(
    poses=mpnn_designs3,
    prefix="boltz_3",
    msa_setting="empty", 
    options="--output_format pdb",
    overwrite=True
)

# ------------------------------------------------------------ # Plotting # ------------------------------------------------------------

# you may plot some important scores. 

from protflow.utils.plotting import violinplot_multiple_cols
violinplot_multiple_cols(
    dataframe=predicted_proteins3.df,
    cols=[
        "boltz_3_confidence_score",
        "boltz_3_ptm",
        "boltz_3_complex_plddt",
        "boltz_3_complex_pde",
    ],
    y_labels=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz complex pLDDT",
        "Boltz complex PDE (Å)",
    ],
    titles=[
        "Boltz confidence",
        "Boltz pTM",
        "Boltz pLDDT",
        "Boltz PDE",
    ],
    out_path=os.path.join(
        predicted_proteins3.plots_dir,
        "boltz_metrics3.png"
    )
)

# ------------------------------------------------------------ # Filtering # ------------------------------------------------------------

predicted_proteins3.filter_poses_by_value(
    score_col="boltz_3_confidence_score",
    value=0.9,  # example cutoff 
    operator=">=",
    prefix="boltz_confidence",
    plot=True
)

predicted_proteins3.filter_poses_by_value(
    score_col="boltz_3_ptm",
    value=0.9,  # example cutoff 
    operator=">=",
    prefix="boltz_ptm",
    plot=True
)

predicted_proteins3.filter_poses_by_value(
    score_col="boltz_3_complex_plddt",
    value=0.9,  # example cutoff 
    operator=">=",
    prefix="boltz_plddt",
    plot=True
)


# and now compare the scores of the first and second round of Boltz predictions, to see if the relaxed structures have improved scores.?? 
# or just compare rmsd to the rfd3 structures?

from protflow.metrics.rmsd import BackboneRMSD

rmsd_calculator = BackboneRMSD(jobstarter=gpu_jobstarter)

rmsd_calculator.run(
    poses=predicted_proteins3,
    prefix="boltz2_vs_boltz3_rmsd",
    ref_col="boltz_2_location" 
)

print(
    predicted_proteins3.df[
        ["poses_description", "boltz2_vs_boltz3_rmsd_rmsd"]
    ]
)

''' 
from protflow.utils.plotting import violinplot_multiple_cols

predicted_proteins3.filter_poses_by_value(
    score_col="boltz2_vs_boltz3_rmsd_rmsd",
    value=1.0,
    operator="<",
    prefix="boltz2_vs_boltz3_rmsd_filter3",
    plot=True
)
'''

from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

sap_protocol = "cm15_data/runners_auxiliary_scripts/sasa_sap_total.xml"

s_options = f"-parser:protocol {sap_protocol} -beta"

scored_poses30 = rosetta.run(
    poses=predicted_proteins3,
    prefix="score_1110",
    nstruct=1,
    options=s_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses30.df.columns.tolist())
print(scored_poses30.df.head())

##################
from protflow.tools.rosetta import Rosetta

rosetta = Rosetta(
    jobstarter=rosetta_jobstarter,
    fail_on_missing_output_poses=True
)

bun_protocol = "cm15_data/runners_auxiliary_scripts/bun_packstat.xml"

b_options = f"-parser:protocol {bun_protocol} -beta"

scored_poses3 = rosetta.run(
    poses=scored_poses30,
    prefix="score_1111",
    nstruct=1,
    options=b_options,
    pose_options="rosetta_options",
    rosetta_application="rosetta_scripts.default.linuxgccrelease"
)

print(scored_poses3.df.columns.tolist())
print(scored_poses3.df.head())

from protflow.utils.plotting import violinplot_multiple_cols
import os

scored_poses3.df["score_per_residue3"] = (
    scored_poses3.df["score_1110_total_score"]
    / scored_poses3.df["rfdiffusion3_num_residues"]
)

scored_poses3.df["total_sasa_per_residue3"] = (
    scored_poses3.df["score_1110_total_sasa_sasa"]
    / scored_poses3.df["rfdiffusion3_num_residues"]
)

scored_poses3.df["sap_per_residue3"] = (
    scored_poses3.df["score_1110_total_sap_sap_score"]
    / scored_poses3.df["rfdiffusion3_num_residues"]
)

violinplot_multiple_cols(
    dataframe=scored_poses3.df,
    cols=[
        "score_1110_sap_score",
        "score_1110_cleavage_sasa_sasa",
        "score_1110_total_sasa_sasa",
        "score_1110_total_score",
        "score_per_residue3",
        "total_sasa_per_residue3",
        "sap_per_residue3",
        "score_1111_buns_foo",
        "score_1111_packstat_foo",
    ],
    y_labels=[
        "SAP score",
        "Cleavage-site SASA",
        "Total SASA",
        "Total Rosetta energy",
        "Average Rosetta energy per residue",
        "SASA per residue",
        "Sap",
        "Buried unsatisfied sc H-bonds",
        "PackStat",
    ],
    titles=[
        "Helix SAP",
        "Cleavage-site SASA",
        "Total SASA",
        "Total energy",
        "Energy/residue",
        "SASA/residue",
        "SAP/residue",
        "Buried unsatisfied H-bonds",
        "PackStat",
    ],
    out_path=os.path.join(
        scored_poses3.plots_dir,
        "scores_sasa_sap_energy_buns_packstat3.png"
    )
)

## or filter by a certain cutoff, e.g. SAP score < 1.0, and plot the distribution of SAP scores before and after filtering.

from protflow.utils.plotting import violinplot_multiple_cols

scored_poses3.filter_poses_by_value(
    score_col="score_per_residue3",
    value=-1.0,
    operator="<=",
    prefix="score_per_res_filter3",
    plot=True
)


scored_poses3.filter_poses_by_value(
    score_col="score_1111_buns_foo",
    value=0.5,
    operator="<=",
    prefix="score_buns_filter3",
    plot=True
)

scored_poses3.filter_poses_by_value(
    score_col="score_1111_packstat_foo",
    value=0.65,
    operator=">=",
    prefix="packstat_filter3",
    plot=True
)


# rank by lowest sap = less exposed hydrophobic surface

sap_ranked3 = scored_poses3.filter_poses_by_rank(
    n=3,
    score_col="score_1110_sap_score",
    prefix="sap_ranked3",
    plot=True,
)

print(sap_ranked3.df[
    ["poses", "rfdiffusion3_description", "score_1110_sap_score", "score_1110_cleavage_sasa_sasa"]
].sort_values("score_110_sap_score").head(20))

""" 
top3_path = os.path.join(
    OUT_DENOVO,
    "top3_pdbs.txt"
)

sap_ranked3.df["poses"].head(3).to_csv(
    top3_path,
    index=False,
    header=False
)

"""

# ------------------------------------------------------------
# Plotting some final metrics of the top structures to inspect their distributions
# ------------------------------------------------------------

from protflow.utils.plotting import violinplot_multiple_cols
import os

violinplot_multiple_cols(
    dataframe=scored_poses3.df,
    cols=[
        "rfdiffusion3_num_residues",
        "mpnn_design_3_overall_confidence",
        "boltz_3_confidence_score",
        "boltz_3_complex_plddt",
        "score_1110_sap_score",
        "score_1110_cleavage_sasa_sasa",
        "score_1111_buns_foo",
        "score_1111_packstat_foo",
        "score_per_residue3",
    ],

    y_labels=[
        "Num Residues",
        "MPNN confidence",
        "Boltz confidence",
        "Boltz complex pLDDT",
        "SAP score",
        "Cleavage-site SASA",
        "Buried unsatisfied sc H-bonds",
        "PackStat",
        "Average Rosetta energy per residue",
    ],

    titles=[
        "Number of residues",
        "MPNN confidence",
        "Boltz confidence",
        "Boltz pLDDT",
        "Helix SAP",
        "Cleavage-site exp",
        "Buried unsatisfied H-bonds",
        "PackStat",
        "Energy/residue",
    ],
    
    out_path=os.path.join(
        scored_poses3.plots_dir,
        "final_metric2.png"
    )
)
