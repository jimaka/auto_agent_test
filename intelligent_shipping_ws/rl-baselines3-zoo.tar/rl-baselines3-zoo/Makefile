LINT_PATHS = *.py tests/ scripts/ rl_zoo3/ hyperparams/python/*.py docs/conf.py

# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent --color=yes

mypy:
	mypy ${LINT_PATHS} --install-types --non-interactive

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# Ship3DOF 专用镜像
docker-ship3dof:
	./scripts/build_docker_ship3dof.sh all

docker-ship3dof-remote:
	./scripts/build_docker_ship3dof.sh remote

docker-isaac-ssh:
	./scripts/build_docker_ship3dof.sh isaac-ssh

# Kubernetes 部署
k8s-apply:
	./scripts/k8s_deploy.sh apply

k8s-train:
	./scripts/k8s_deploy.sh train

k8s-status:
	./scripts/k8s_deploy.sh status

# SSH 远程开发容器
ssh-container-create:
	./scripts/create_ssh_container.sh create

ssh-container-status:
	./scripts/create_ssh_container.sh status

# 多 GPU 并行训练
multi-gpu:
	./scripts/ship_3dof/run_multi_gpu.sh

multi-gpu-smoke:
	./scripts/ship_3dof/run_multi_gpu.sh --smoke

multi-gpu-docker:
	./scripts/ship_3dof/run_multi_gpu_docker.sh

# PyPi package release
release:
	# rm -r build/* dist/*
	python -m build -s
	python -m build -w
	twine upload dist/*

# Test PyPi package release
test-release:
	# rm -r build/* dist/*
	python -m build -s
	python -m build -w
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks doc spelling docker type pytest ssh-container-create ssh-container-status multi-gpu multi-gpu-smoke multi-gpu-docker docker-isaac-ssh
