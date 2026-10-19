# Kubernetes deployment

## Minikube

```bash
minikube start
docker build -t heart-disease-mlops:latest .
minikube image load heart-disease-mlops:latest
kubectl apply -f k8s/
kubectl get pods -n heart-mlops
kubectl get svc -n heart-mlops
minikube tunnel
```

## Verify

```bash
kubectl rollout status deployment/heart-api -n heart-mlops
kubectl logs deployment/heart-api -n heart-mlops
kubectl get endpoints -n heart-mlops
```

API:

```text
http://<EXTERNAL-IP>/health
http://<EXTERNAL-IP>/metrics
```

Prometheus:

```bash
kubectl port-forward svc/prometheus 9090:9090 -n heart-mlops
```

Grafana:

```bash
kubectl get svc grafana -n heart-mlops
minikube service grafana -n heart-mlops
```
