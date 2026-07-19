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
  
### Setting up Local Files
  
It is quite possible that you will be unable to open some file in the codebase because a "file was not found". In this case, you should move a copy of the file to the cluster, and then update the path with the configs to point to the cluster. First check to see if the file is present in the `shared` file directory, where all the files are located. That is, run `cd /share/data/school_choice/` and see if the file lives in there.
  
For example, you will have to copy the dropout rates file locally from the Cluster's shared directory, which you can do with
  
```
Note: Only run these two commands after running the two commands later in this section.
$ mkdir ~/sfusd/sfusd-local-data/Data/Precomputed
$ cp /share/data/school_choice/Data/Precomputed/student_program_distances_1819.csv ~/sfusd/sfusd-local-data/Data/Precomputed/student_program_distances_dropoptout_1819.csv
```
  
Otherwise, you should add the file locally. Any Unix system comes with SCP, which will allow you to move files to folders onto the clusters. If you installed putty for windows, it should also come with SCP. Using SCP is fairly simple -- it's mostly cd'ing to the directory that contains the files you want to move, and running the right command. For example, I ran this command to move some local data files I had, into a folder I called `sfusd-local-data` in my `sfusd` directory.

That is, on my local machine I went to the Dropbox SFUSD folder, went into the starter-data directory, and ran
  
```
$ cd ~/Dropbox/SFUSD/starter-data  
$ scp * $USER@soal-cluster.stanford.edu:~/sfusd/sfusd-local-data
```

For some more tricks, visit https://www.simplified.guide/ssh/copy-file.

### Setting up your Config
  
The first time you run any entry point (e.g. `uv run python run_custom_config.py --config-path ...`), the `Configerator` automatically populates your personal `configs/<sunet-id>.config.yaml` by merging `base_config.yaml` with the environment-specific path config. 
  
Then, you can go into the `configs` subdirectory in `simulator_engine`, and you should see `<sunet-id>.config.yaml`. Open this file, and change your config to replace my username (hguru) with your username in the paths below "Local paths" (if you're wondering how to edit files on the cluster, look below at the Development Workflow section, to learn how to write code directly on the cluster).
  
# Life after the setup
  
## Development workflow
  
I recommend both running code and developing on the cluster. These days there are fabulous tools in modern IDEs to develop remotely. I recommend using VS Code. You can see the guide for developing on a remote server here: https://code.visualstudio.com/docs/remote/ssh. Make sure to add extensions once you've connected to the remote server, such as the Python extension.
