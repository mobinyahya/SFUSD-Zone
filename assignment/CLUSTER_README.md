# Setting up for the first time on Cluster (Work in Progress; Ignore for Now)

## Short Guide for Setting Up

Please do read the full guide below, but if you are in need of a quick refresher or as a cheat sheet, here is a summary of the major steps below.

1. Connect to [Stanford VPN](https://uit.stanford.edu/service/vpn).
2. Connect to [SOAL](https://5harad.com/soal-cluster/) via SSH by running `ssh <sunet-id-here>@soal-cluster.stanford.edu`.
3. Clone this repository into a desired directory, by running `git clone https://github.com/klmentzer/sfusd-project.git` into that directory.
4. Install uv (user-local, no root) and run `uv sync`; see "Managing Dependencies" below.

## Long Guide for Setting Up

This project requires computational power at certain stages. As a result, it is suggested that you run the code in [SOAL](https://5harad.com/soal-cluster/), the cluster hosted by Sharad Goel's research group. In addition to the added computational power of the cluster, we also have all data necessary for the project in the cluster.

### Running & Setting up on the cluster

If you need access to SOAL but don't have access, speak to Itai -- he can speak to Sharad and create a local environment for you in the cluster. 

Before connecting to the cluster, however, you need to run through Stanford's proxy. For instructions about how to use this proxy, follow [these instructions](https://uit.stanford.edu/service/vpn). I am currently developing with Full Traffic; just note that all of your packets will be forwarded through Stanford servers, so if you are doing something sensitive or personal non-related to Stanford, it's a good idea to make sure that you quit out of the Cisco client and ensure that the VPN isn't running anymore.

Once you have configured the VPN, you can connect to the cluster using SSH. For most machines, this means doing something like typing, 

```{bash}
ssh <sunet-id-here>@soal-cluster.stanford.edu
```

into your terminal if you are on a UNIX system (Mac, Linux, etc.) or into any SSH client of your choice in Windows ([putty](https://www.putty.org/), etc.).

At this point, you should be prompted for your password (your SUNET password), and some form of Duo two-factor authentication. Once this has finished, you should be connected to the cluster. That is, you should have a command line open with

```{bash}
<sunet-id>@soal-?:~$
```
Next, you should make a folder for your `sfusd` project related files. One of the subdirectories in this folder will acutally be this repo! You can do this by running the following commands:

```{bash}
$ mkdir sfusd
$ cd sfusd
$ git clone https://github.com/klmentzer/sfusd-project.git
```

You will be asked for your Github credentials, and when provided, it will add a copy of the repo in the subdirectory, `sfusd-project`.

### Managing Dependencies

This project manages dependencies with **[uv](https://docs.astral.sh/uv/)**,
pinned in `pyproject.toml` + `uv.lock`. uv installs as a user-local static
binary, so it needs no root on the cluster. To get started:

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then make sure
   `~/.local/bin` is on your PATH (the installer adds it; restart your shell or
   `. ~/.bashrc`).
2. From the repository root, run `uv sync`. This creates a `.venv/` and installs
   every pinned dependency from `uv.lock`.
3. Run project commands with `uv run <cmd>` (e.g.
   `uv run python run_custom_config.py ...`), or activate the env with
   `source .venv/bin/activate`.

Everyone who runs `uv sync` gets the identical pinned environment, so
dependencies stay synchronized across users. To add an external dependency, run
`uv add <package-name>`.

### Slurm assignment runs

Generate a resolved plan and scripts without contacting Slurm:

```bash
uv run python -m assignment.slurm generate --config assignment/configs/kumar.config.yaml
```

Submit the generated job graph directly:

```bash
uv run python -m assignment.slurm submit --config assignment/configs/kumar.config.yaml
```

The launcher creates one one-core job for every subconfig and iteration pair.
When `export-aggregate-metrics` is true, it also creates one dependent metrics
job per subconfig. `export-local-metrics` adds school, ZIP code, and attendance
area CSVs to the citywide report. Plans, worker scripts, and logs are written
under `<paths.assignment-folder>/slurm/`; aggregate CSV updates are locked and
safe to retry. All jobs use Slurm account and partition `soal`.
  
### Data Files

Application source data is read directly from `/soalnas/share/data/school_choice/`.
Do not create per-user copies or change generated configs to point into a home
directory. If a required file is missing, add it to the documented shared
location in `DATA_FOLDERS.md`.

### Setting up your Config
  
The first time you run an entry point, `Configerator` populates your personal
`configs/<sunet-id>.config.yaml` from `base_config.yaml` plus the local output
path config. Input data always comes from the strict top-level `data` scenario;
there is no hostname-based input-path selection.
  
# Life after the setup
  
## Development workflow
  
I recommend both running code and developing on the cluster. These days there are fabulous tools in modern IDEs to develop remotely. I recommend using VS Code. You can see the guide for developing on a remote server here: https://code.visualstudio.com/docs/remote/ssh. Make sure to add extensions once you've connected to the remote server, such as the Python extension.
