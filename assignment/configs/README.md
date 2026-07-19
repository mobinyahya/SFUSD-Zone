# Setting Up Your Config File

Make all your configuration changes in your personal config file, `<YOUR-COMPUTER-USERNAME>.config.yaml`.
If you do not see a file like this in `configs/`, it is generated automatically the first time you run any entry point (e.g. `uv run python run_custom_config.py --config-path <config>.yaml`).
DO NOT make changes to `base_config.yaml`.

This file will contain both your local file paths and non-policy related simulation options (like whether or not to use a utility model).

To select which policies to run, add the policy configs to the subconfig section of your config. 
For example,
```yaml
subconfigs:
  - zones+reserves
  - real_match
```

## Troubleshooting

The first time you generate your config, you will likely need to adjust the filepaths in your personal config.
If you get path related errors, check the file that raised the error in your config and adjust accordingly.

If you are getting validation errors loading your config (missing required values or unexpected values), try re-generating your config from the base config 
by changing the name of your config file or deleting it.
(Changing the name is recommended for easier updating of local paths.)


## Generating New Policies

The easiest way to create new policies is to copy an old policy and adjust the desired values accordingly.
Note that a policy can contain multiple different zones (i.e., multiple different home based plans) but will keep the other policy details (like using guardrails) constant.
Paths to the appropriate zone files go in your config file, then call the name under "policies" in the policy config.