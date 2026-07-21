#!/bin/bash
# Kubernetes 部署脚本
# 用法:
#   ./scripts/k8s_deploy.sh build              # 构建并推送镜像
#   ./scripts/k8s_deploy.sh apply              # 部署远程桌面环境
#   ./scripts/k8s_deploy.sh train              # 提交训练 Job
#   ./scripts/k8s_deploy.sh status             # 查看部署状态
#   ./scripts/k8s_deploy.sh delete             # 删除所有资源
#   ./scripts/k8s_deploy.sh port-forward       # 本地端口转发（无需 Ingress）
#
# 环境变量:
#   REGISTRY     - 镜像仓库地址 (默认: 本地镜像，不推送)
#   NAMESPACE    - K8s 命名空间 (默认: ship3dof)
#   GPU_COUNT    - GPU 数量 (默认: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s"
NAMESPACE="${NAMESPACE:-ship3dof}"
REGISTRY="${REGISTRY:-}"
GPU_COUNT="${GPU_COUNT:-1}"

ACTION="${1:-apply}"

build_and_push() {
    echo ">>> 构建 Docker 镜像..."
    "${SCRIPT_DIR}/build_docker_ship3dof.sh" all

    if [[ -n "${REGISTRY}" ]]; then
        echo ">>> 推送镜像到 ${REGISTRY}..."
        docker tag ship3dof-remote:latest "${REGISTRY}/ship3dof-remote:latest"
        docker tag ship3dof:latest "${REGISTRY}/ship3dof:latest"
        docker push "${REGISTRY}/ship3dof-remote:latest"
        docker push "${REGISTRY}/ship3dof:latest"

        # 更新 kustomization 镜像地址
        cd "${K8S_DIR}"
        kustomize edit set image "ship3dof-remote=${REGISTRY}/ship3dof-remote:latest"
    fi
}

apply_resources() {
    echo ">>> 部署到命名空间 ${NAMESPACE}..."
    kubectl apply -k "${K8S_DIR}"
    echo ""
    echo ">>> 等待 Pod 就绪..."
    kubectl -n "${NAMESPACE}" rollout status deployment/ship3dof-remote --timeout=300s || true
    show_status
}

submit_train_job() {
    echo ">>> 提交训练 Job..."
    kubectl create -f "${K8S_DIR}/job-train.yaml" -n "${NAMESPACE}"
    echo ">>> 查看 Job: kubectl -n ${NAMESPACE} get jobs -l app.kubernetes.io/component=training"
}

submit_autotune_job() {
    echo ">>> 提交自主调参 Job（需 2 GPU）..."
    kubectl create -f "${K8S_DIR}/job-autotune.yaml" -n "${NAMESPACE}"
    echo ">>> 查看 Job: kubectl -n ${NAMESPACE} get jobs -l app.kubernetes.io/component=autotune"
}

submit_multi_gpu_job() {
    echo ">>> 提交多 GPU 并行训练 Job（需 2 GPU）..."
    kubectl create -f "${K8S_DIR}/job-multi-gpu.yaml" -n "${NAMESPACE}"
    echo ">>> 查看 Job: kubectl -n ${NAMESPACE} get jobs -l app.kubernetes.io/component=multi-gpu-training"
}

show_status() {
    echo ""
    echo "=============================================="
    echo "  Ship3DOF K8s 部署状态"
    echo "=============================================="
    kubectl -n "${NAMESPACE}" get pods,svc,pvc,ingress 2>/dev/null || true
    echo ""
    echo "访问方式:"
    echo "  1. Ingress:  配置域名后访问 https://ship3dof.example.com/vnc"
    echo "  2. 端口转发: ./scripts/k8s_deploy.sh port-forward"
    echo "  3. NodePort: kubectl -n ${NAMESPACE} patch svc ship3dof-remote -p '{\"spec\":{\"type\":\"NodePort\"}}'"
    echo "=============================================="
}

port_forward() {
    echo ">>> 启动本地端口转发..."
    echo "  远程桌面:    http://localhost:6080/vnc.html"
    echo "  Jupyter:     http://localhost:8888"
    echo "  TensorBoard: http://localhost:6006"
    echo "  SSH:         ssh developer@localhost -p 2222"
    echo ""
    kubectl -n "${NAMESPACE}" port-forward svc/ship3dof-remote \
        6080:6080 8888:8888 6006:6006 2222:22
}

delete_resources() {
    echo ">>> 删除所有 Ship3DOF 资源..."
    kubectl delete -k "${K8S_DIR}" --ignore-not-found
    echo ">>> 资源已删除（PVC 数据保留，需手动删除: kubectl -n ${NAMESPACE} delete pvc --all）"
}

case "${ACTION}" in
    build)
        build_and_push
        ;;
    apply)
        apply_resources
        ;;
    train)
        submit_train_job
        ;;
    autotune)
        submit_autotune_job
        ;;
    multi-gpu|mgpu)
        submit_multi_gpu_job
        ;;
    status)
        show_status
        ;;
    port-forward|pf)
        port_forward
        ;;
    delete)
        delete_resources
        ;;
    all)
        build_and_push
        apply_resources
        ;;
    *)
        echo "用法: $0 {build|apply|train|autotune|multi-gpu|status|port-forward|delete|all}"
        exit 1
        ;;
esac
