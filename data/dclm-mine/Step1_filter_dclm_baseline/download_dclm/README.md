### Step 1: Download DCLM baseline

# allocates folder: /data/horse/ws/rehe951g-dcagent
ws_allocate dcagent 90 

# generate list of urls with files to be downloaded
python make_dclm_urls.py 

# downloads all of them as intended with high-bandwidth datamover nodes
dtwget --account=p_agents_finetuning -P /data/horse/ws/rehe951g-dcagent/dclm-baseline/ -x --no-host-directories --cut-dirs=5 -i /home/rehe951g/ot-agent/data/dclm-mine/Step1_filter_dclm_baseline/download_dclm/urls.txt