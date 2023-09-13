install:
	pip install --upgrade pip &&\
		pip install --no-cache-dir -r requirements.txt

test:
	#python -m pytest -vv test_application.py

lint:
	pylint --disable=R,C app.py

all: install lint test 
