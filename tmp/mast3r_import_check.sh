cd /hdd3/kykt26/code/mast3r
pwd
ls -d mast3r dust3r
conda run -n mast3r python -c "import os,sys; print(os.getcwd()); print(sys.path[:3]); import mast3r; print('IMPORT_OK')"