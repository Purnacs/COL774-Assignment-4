all: v2

v2: Ass4_Ver2.py
	python3 Ass4_Ver2.py ../Dataset

v3: Ass4_Ver3.py
	python3 Ass4_Ver3.py ../Dataset	

data_process: Data_processing.py
	python3 Data_processing.py ../Dataset

HPC_Auto: Ass4_Ver3.py
	python3 Ass4_Ver3.py ../Dataset
	git remote set-url origin git@github.com:Purnacs/COL774-Assignment-4.git
	git add .
	git commit -m "HPC Auto Commit"
	git push origin HPC_Automate

	
