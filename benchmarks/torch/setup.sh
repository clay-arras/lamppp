git clone https://github.com/pytorch/pytorch.git
cd pytorch

cd benchmarks/operator_benchmark
cp ../../../all_tests.py pt/all_tests.py

python -m venv .venv
.venv/bin/python -m pip install -r ../../../requirements.txt
cd pt_extension && ../.venv/bin/python setup.py install


