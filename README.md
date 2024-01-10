# TransKeys
Transformer based model to detect aws keys

### Step 1

```pip3 install -r requirements.txt```


### Step 2

```gunicorn -k uvicorn.workers.UvicornWorker inference_code:app --bind 0.0.0.0:8000 --workers 4 --max-requests 1000 --max-requests-jitter 50```

### Note:
```Please use Linux/Unix based platform```