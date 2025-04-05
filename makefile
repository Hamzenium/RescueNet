# Makefile

env:
	python3 -m venv env

install: env
	. env/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	. env/bin/activate && python train.py

evaluate:
	. env/bin/activate && python inference.py

	
plot:
	. env/bin/activate && python plot.py

clean:
	rm -rf __pycache__ ./results ./logs env
